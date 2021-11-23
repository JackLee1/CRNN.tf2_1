import tensorflow as tf
import os
import cv2 
import numpy as np
import shutil
from tensorflow import keras

class ImageCallback(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, folder, dataset, stn_model, require_coords, row=8, count=2, point=6):
        super(ImageCallback, self).__init__()
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        self.folder = folder
        self.dataset = dataset
        self.stn_model = stn_model
        self.require_coords = require_coords
        self.row=row
        self.count = count
        self.point = point
    
    def get_predict_point(self, transform_mat):
        """
        transform_mat: (batch, 6)
        """

        if self.point == 4:
            zeros=tf.zeros_like(transform_mat)[:,:1]
            # affine_transforms=(batch, 6)
            transform_mat = tf.concat([transform_mat[:,0:1], zeros, transform_mat[:,1:2], zeros, transform_mat[:,2:4]],1)
        
        #print("get_predict_point : transform_mat  ",end = "")
        #print(type(transform_mat[1]), transform_mat.shape,transform_mat) 
        #tf.print(tf.shape(transform_mat))
        transform_mat = np.array(transform_mat)
        #print("\n\nafter\n\n")
        #print(type(transform_mat[1]), transform_mat.shape,transform_mat)

        transform_mat = transform_mat.reshape((-1, 2, 3))
        my_coord = np.array([[
            [-1,-1,1],
            [ 1, 1,1],
            [ 1,-1,1],
            [-1, 1,1]
        ]])
        my_coord = my_coord.transpose((0,2,1))
        new_coord = np.matmul(transform_mat, my_coord)
        new_coord = new_coord.transpose((0,2,1))
        return new_coord

    def coord_to_int(self, coords, imgshape):
        b, ih, iw = imgshape[:3]
        n_points=coords.shape[-1] // 2
        ncoords = (coords + 1.0) / 2.0 * np.array([[iw, ih]*n_points])
        ncoords = ncoords.astype(np.int32).reshape((b, -1))
        return ncoords

    def on_epoch_end(self, epoch, logs={}):
        for i, (images,labels) in enumerate(self.dataset, 1):
            if self.require_coords:
                label, coords=labels
            stn_result, transform_mat = self.stn_model(images, training=False)
            # process origin image
            images = images.numpy()[:self.row]
            images = (images*255.).astype(np.uint8)
            transform_mat = transform_mat.numpy()[:self.row]

            #print("on_epoch_end : transform_mat  ",end = "")
            #print(type(transform_mat[1]), transform_mat.shape,transform_mat) 
            #tf.print(tf.shape(transform_mat))

            pcoords = self.coord_to_int(self.get_predict_point(transform_mat), images.shape)
            if self.require_coords: 
                gcoords = self.coord_to_int(coords.numpy()[:self.row], images.shape)
            n_points = pcoords.shape[-1] // 2
            for ii in range(len(images)):
                img = images[ii].copy()
                for iii in range(n_points):
                    if self.require_coords: 
                        images[ii] = cv2.circle(img, tuple(gcoords[ii,2*iii:2*(iii+1)]), 3, (int(127+128/4*iii), 0, 0), -1)
                    images[ii] = cv2.circle(img, tuple(pcoords[ii,2*iii:2*(iii+1)]), 3, (0, 0, int(127+128/4*iii)), -1)
                    
            images = np.vstack(images)
            # process stn_result
            stn_result = stn_result.numpy()[:self.row]
            stn_result = (stn_result*255.).astype(np.uint8)
            stn_result = np.vstack(stn_result)
            h, w = stn_result.shape[:2]
            w = int(w * images.shape[0] / h)
            h = int(images.shape[0])
            stn_result = cv2.resize(stn_result, (w, h))
            stn_result[:,0]=255
            # total image
            filename = f'epoch_{epoch}_{i}.png'
            show_result = np.concatenate([images, stn_result], axis=1)
            cv2.imwrite(os.path.join(self.folder, filename), show_result[...,::-1])
            if i == self.count: break


class ModelWeight(keras.callbacks.Callback):
    def __init__(self, layername):
        super().__init__()
        self.layername=layername
    def on_epoch_end(self, epoch, logs=None):
        weights=self.model.get_layer(self.layername).get_weights()
        print(tf.reshape(weights[1], (-1)))

