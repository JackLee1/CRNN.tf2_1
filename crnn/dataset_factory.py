import os
import re
from imgaug import augmenters as iaa
import imgaug as ia
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import argparse
import yaml
import shutil
import cv2
from pathlib import Path

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.data.TextLineDataset):
    def __init__(self, filename, **kwargs):
        self.dirname = os.path.dirname(filename)
        super().__init__(filename, **kwargs)

    def parse_func(self, line):
        raise NotImplementedError

    def parse_line(self, line):
        line = tf.strings.strip(line)
        img_relative_path, label, coord = self.parse_func(line)
        img_path = tf.strings.join([self.dirname, os.sep, img_relative_path])
        return img_path, label, coord


class SimpleDataset(Dataset):
    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path, label = splited_line[0], splited_line[1]
        coord = tf.strings.to_number(splited_line[2:])
        coord = coord[:4]
        return img_relative_path, label, coord

class DatasetBuilder:
    def __init__(self, table_path, img_shape=(32, None, 3), max_img_width=300, require_coords=False):
        # map unknown label to 0
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), 0)
        self.img_shape = img_shape
        self.require_coords = require_coords

        if img_shape[1] is None:
            self.max_img_width = max_img_width
            self.preserve_aspect_ratio = True
        else:
            self.preserve_aspect_ratio = False

    @property
    def num_classes(self):
        return self.table.size()

    def _parse_annotation(self, path):
        with open(path) as f:
            line = f.readline().strip()
        if re.fullmatch(r'.+\.\w+ .+', line):
            return SimpleDataset(path)
        else:
            raise ValueError('Unsupported annotation format')

    def _concatenate_ds(self, ann_paths):
        datasets = [self._parse_annotation(path) for path in ann_paths]
        concatenated_ds = datasets[0].map(datasets[0].parse_line)
        for ds in datasets[1:]:
            ds = ds.map(ds.parse_line)
            concatenated_ds = concatenated_ds.concatenate(ds)
        return concatenated_ds

    @tf.function
    def _decode_img(self, filename, label, coord):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        if self.preserve_aspect_ratio:
            img_shape = tf.shape(img)
            scale_factor = self.img_shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_shape[1]
        img = tf.image.resize(img, (self.img_shape[0], img_width)) # / 255.0
        img = tf.cast(img, 'uint8')
        return img, label, coord

    @tf.function
    def _filter_img(self, img, label, coord):
        img_shape = tf.shape(img)
        return img_shape[1] < self.max_img_width

    @tf.function
    def _tokenize(self, imgs, labels, coord):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens, coord

    @tf.function
    def _return_shape(self, img, label, coord):
        """
        img: (w, h, 3), single image
        label: (n,), character label
        coord: (4,), boundingbox coordianate in [-1,1]
        """
        shape = tf.shape(img)
        return img, label, coord, shape
    
    @tf.function
    def _translate(self, imgs, labels, coord, old_shapes):
        """
        imgs: (b, w, h, 3), batch image
        label: (b, n), batch character label
        coord: (b, 4), batch boundingbox coordianate in [-1,1]
        shapes: (b, 3), batch image origin shape
        """
        old_shapes=tf.cast(old_shapes, tf.float32)
        aug_shape=tf.cast(tf.shape(imgs), tf.float32)
        x_shift = aug_shape[2] - old_shapes[:, 1:2]
        y_shift = tf.zeros_like(x_shift)
        shift = tf.concat([x_shift, y_shift], axis=1) / 2.0
        newimg = tfa.image.translate(imgs, shift)
        
        reduce_ratio = old_shapes[:,:2] / tf.expand_dims(aug_shape[1:3], axis=0)
        reduce_ratio = tf.concat([reduce_ratio,reduce_ratio],axis=-1)[...,::-1]
        
        if self.require_coords:
            coord = coord * reduce_ratio # [-1,1]
        # error_idx = tf.reduce_all(tf.abs(coord) <= 1.0, axis=1)

        return newimg, labels, coord

    @tf.function
    def _add_coordinate(self, imgs, labels, coords):
        x1 = coords[...,0:1]
        y1 = coords[...,1:2]
        x2 = coords[...,2:3]
        y2 = coords[...,3:4]
        right_top=tf.concat([x2,y1], axis=-1)
        left_bot=tf.concat([x1,y2], axis=-1)
        coord4s=tf.concat([coords, right_top, left_bot], axis=-1)
        return imgs, labels, coord4s

    def create_aug_env(self, p=0.5):
        seq = iaa.Sequential([
            iaa.Sometimes(1.0, 
                iaa.Affine(
                    scale={"x": (0.8, 1.3), "y": (0.6, 1.2)},
                    translate_percent={"x": (-0.3, 0.3), "y": (-0.05, 0.05)},
                    shear={"x": (-10, 10), "y": (-5, 5)},
                    rotate=(3,-3)#
                ),    
            ),
            iaa.Sometimes(p, 
                iaa.OneOf([
                    iaa.CoarseDropout(p=(0.05, 0.15), size_percent=0.7),
                    iaa.SaltAndPepper(p=0.04),
                ])
            ),
            iaa.SomeOf((0, 2), [
                iaa.ChannelShuffle(0.3),
                # iaa.AdditiveGaussianNoise(scale=0.05*255),
                iaa.GammaContrast(gamma=(0.5,1.4)),
                iaa.Multiply((0.7, 1.2)),
                iaa.pillike.EnhanceColor(factor=(0.5, 2.0))
            ])
        ], random_order=True)
        return seq

    @tf.function
    def dtype_transform(self, a,b,c):
        a = tf.cast(a, 'float32') / 255.0
        return a, b, c
    
    def __call__(self, ann_paths, batch_size, is_training):
        ds = self._concatenate_ds(ann_paths)
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img, AUTOTUNE)
        if self.preserve_aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)
            ds = ds.map(self._return_shape)
            ds = ds.padded_batch(batch_size, drop_remainder=is_training)
            ds = ds.map(self._translate)
            if self.require_coords: ds = ds.map(self._add_coordinate)
        else:
            ds = ds.batch(batch_size, drop_remainder=is_training)

        seq = self.create_aug_env()
        def augmentation(imgs, labels, coords):
            def do_augmentation(a_img, ori_coords):
                if not self.require_coords:
                    aa_imgs=seq(images=a_img)
                    return aa_imgs, ori_coords
                else:
                    b, h, w = a_img.shape[:3]
                    ori_coords = (ori_coords+1.0)/2.0
                    ori_coords = np.reshape(ori_coords, (b, -1, 2)) * np.array([[[w, h]]])
                    ori_coords = ori_coords.astype(np.int32)
                    ori_coords = [
                        ia.augmentables.kps.KeypointsOnImage.from_xy_array(ori_coords[k], shape=a_img[k].shape) 
                        for k in range(len(ori_coords))]
                    aa_imgs, aa_coords=seq(images=a_img, keypoints=ori_coords)
                    aa_coords=[kp.to_xy_array() for kp in aa_coords]
                    
                    aa_coords=np.array(aa_coords) / np.array([[[w, h]]])
                    aa_coords=np.reshape(aa_coords, (b, -1))
                    aa_coords=tf.cast((aa_coords-0.5)*2.0, tf.float32)
                    return aa_imgs, aa_coords
            aug_imgs, aug_points = tf.numpy_function(do_augmentation, inp=[imgs, coords], Tout=[tf.uint8, tf.float32])
            return aug_imgs, labels, aug_points

        ds = ds.map(self._tokenize, AUTOTUNE)
        if is_training: 
            ds = ds.map(augmentation, AUTOTUNE)
        # convert uint8 to float32
        ds = ds.map(self.dtype_transform, AUTOTUNE)

        if self.require_coords: 
            ds = ds.map(lambda a,b,c: (a, (b,c)))
        else: 
            ds = ds.map(lambda a,b,c: (a,(b,)))
        ds = ds.prefetch(AUTOTUNE)
        return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='The config file path.')
    args = parser.parse_args()

    target_path='./data_vis'
    # shutil.rmtree(target_path, ignore_errors=True)
    os.makedirs(target_path, exist_ok=True)
    with args.config.open() as f:
        config = yaml.load(f, Loader=yaml.Loader)['train']
    dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=False)
    train_ds = dataset_builder(config['train_ann_paths'], 16, True)
    val_ds = dataset_builder(config['val_ann_paths'], 16, False)

    # Testing Augmentation for images
    mylist=[]
    for i, (a, b) in enumerate(train_ds, 1):
        a=tf.cast(a*255.0, tf.uint8).numpy()
        a=np.vstack(a)
        mylist.append(a)
        if len(mylist) != 2:
            continue
        else:
            filename=f'{i//2}.png'
            full_img = np.hstack(mylist)
            cv2.imwrite(os.path.join(target_path, filename), full_img)
            mylist=[]
        if i == 8: break

    # Testing Augmentation for single images
    for i, (a,b) in enumerate(val_ds, 1):
        single_img=a[0].numpy()
        single_img=(single_img*255.0).astype(np.uint8)
        break
    aug_env = dataset_builder.create_aug_env()
    col_list=[]
    for c in range(4):
        row_list=[]
        for r in range(8):
            res=aug_env(image=single_img)
            row_list.append(res)
        row_list = np.vstack(row_list)
        col_list.append(row_list)
    col_list = np.hstack(col_list)
    col_list = col_list.astype(np.uint8)
    cv2.imwrite(os.path.join(target_path, 'aug_img.png'),col_list)

    