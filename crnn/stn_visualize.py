import argparse
import pprint
import tensorflow as tf
import yaml
import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt

from dataset_factory import DatasetBuilder
from losses import CTCLoss, LossBox
from metrics import SequenceAccuracy, EditDistance
from models import build_model
from layers.stn import BilinearInterpolation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='The config file path.')
parser.add_argument('--weight', type=str, default='', required=False, help='The saved weight path.')
parser.add_argument('--structure', type=str, default='', required=False, help='The saved structure path.')
parser.add_argument('--iou', type=float, default=0.5, required=False, help='IoU threshold')
args = parser.parse_args()
args.point4=True
if args.structure == '':
    args.structure = os.path.join(os.path.dirname(args.weight), 'structure.h5')


batch_size=16

with open(args.config) as f:
    parse_config = yaml.load(f, Loader=yaml.Loader)
    config = parse_config['eval']
    train_conf = parse_config['train']
pprint.pprint(config)

stn_vis_dir = 'stn_vis'
shutil.rmtree(stn_vis_dir, ignore_errors=True)
os.makedirs(stn_vis_dir, exist_ok=True)
false_path = os.path.join(stn_vis_dir, 'False')
true_path = os.path.join(stn_vis_dir, 'True')
shutil.rmtree(false_path, ignore_errors=True)
shutil.rmtree(true_path, ignore_errors=True)
os.makedirs(false_path, exist_ok=True)
os.makedirs(true_path, exist_ok=True)

dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=args.point4)
ds = dataset_builder(config['ann_paths'], config['batch_size'], False)
val_ds = dataset_builder(train_conf['val_ann_paths'], batch_size, False)
model = tf.keras.models.load_model(args.structure, custom_objects={
    'BilinearInterpolation': BilinearInterpolation
}, compile=False)
model.load_weights(args.weight)

inputs=model.layers[0].input
if args.point4:
    outputs1=model.get_layer('ctc_logits').output
    outputs2=model.get_layer('stn').output
    model = tf.keras.Model(inputs, [outputs1, outputs2])
else:
    outputs1=model.get_layer('ctc_logits').output
    model = tf.keras.Model(inputs, outputs1)
model.summary()

print(model.output_names)
if args.point4:
    loss_dict={ 
        model.output_names[0]: [CTCLoss()],
        model.output_names[1]: [LossBox()] 
    }
    metrics_dict={ model.output_names[0]: [SequenceAccuracy(), EditDistance()] }
else:
    loss_dict=[CTCLoss()]
    metrics_dict=[SequenceAccuracy(), EditDistance()]

model.compile(loss=loss_dict, metrics=metrics_dict)

def get_predict_point(transform_mat):
    """
    transform_mat: (batch, 6)
    """
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

def coord_to_int(coords, imgshape, convert_int=True):
    b, ih, iw = imgshape[:3]
    n_points=coords.shape[-1] // 2
    ncoords = (coords + 1.0) / 2.0 * np.array([[iw, ih]*n_points])
    if convert_int: ncoords = ncoords.astype(np.int32)
    return ncoords.reshape((b, -1))

def calculate_iou(coord1, coord2):
    area1 = (coord1[...,2]-coord1[...,0]) * (coord1[...,3]-coord1[...,1])
    area2 = (coord2[...,2]-coord2[...,0]) * (coord2[...,3]-coord2[...,1])
    inter_x1 = np.maximum(coord1[...,0], coord2[...,0])
    inter_y1 = np.maximum(coord1[...,1], coord2[...,1])
    inter_x2 = np.minimum(coord1[...,2], coord2[...,2])
    inter_y2 = np.minimum(coord1[...,3], coord2[...,3])
    inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
    union = area1 + area2 - inter
    iou = inter / union
    return iou


h_list=[]
w_list=[]
label_list=[]
total=0
acc = 0
for i, (images, (ctc_label, stn_label)) in enumerate(val_ds):    
    ctc_pred, transform_mat = model(images)

    images = images.numpy()
    transform_mat = transform_mat.numpy()
    stn_label = stn_label.numpy()
    images = (images*255.).astype(np.uint8)
    p_coord = coord_to_int(get_predict_point(transform_mat), images.shape)
    g_goord = coord_to_int(stn_label, images.shape)
    g_goordf = coord_to_int(stn_label, images.shape, False)
    iou = calculate_iou(p_coord, g_goord)
    # print(iou)
    for ii in range(len(images)):
        img = images[ii].copy()
        for iii in range(4):
            # images[ii] = cv2.circle(img, tuple(g_goord[ii,2*iii:2*(iii+1)]), 3, (255, 0, 0), -1)
            images[ii] = cv2.circle(img, tuple(p_coord[ii,2*iii:2*(iii+1)]), 3, (0, 0, 255), -1)




    if False:
        image1 = np.vstack(images[:batch_size//2])
        image2 = np.vstack(images[batch_size//2:])
        images = np.hstack([image1, image2])
        
        # total image
        filename = f'image_{i}.png'
        cv2.imwrite(os.path.join(stn_vis_dir, filename), images)
    if True:
        for ii in range(len(images)):
            t_iou = iou[ii]
            str_iou = str(t_iou)[2:4]
            total+=1
            if t_iou > args.iou: 
                acc+=1
                save_path = os.path.join(true_path, f'{i*batch_size+ii}_{str_iou}.png')
            else: save_path = os.path.join(false_path, f'{i*batch_size+ii}_{str_iou}.png')
            label_list.append(int(t_iou > args.iou))
            if i < 100:
                cv2.imwrite(save_path, images[ii,...,::-1])
                print(f'{save_path:40s} {t_iou > args.iou}')

        hw = g_goordf[:,2:4] - g_goordf[:,:2]
        h = hw[:,0]
        w = hw[:,1]

        h_list.append(h)
        w_list.append(w)



# h_list=np.concatenate(h_list)
# w_list=np.concatenate(w_list)
# label_list=np.array(label_list)

# f_points=plt.scatter(x=h_list[label_list==0],y=w_list[label_list==0], marker = 'D', color='r', s=1)
# t_points=plt.scatter(x=h_list[label_list==1],y=w_list[label_list==1], marker = 'D', color='g', s=1)
# plt.legend([f_points, t_points], ['false', 'true'], loc='lower right', scatterpoints=1)

# plt.legend([t_points, f_points], ['True', 'False'])
# plt.savefig(os.path.join(stn_vis_dir, 'area.png'))
print('Total: ', total)
print('Accuracy: ', acc/total)