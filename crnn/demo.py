import argparse
import pprint
import shutil
from pathlib import Path

import tensorflow as tf
import yaml
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from layers.stn import BilinearInterpolation
import glob
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--images', type=str, required=True, help='The image file path.')
parser.add_argument('--structure', type=str, required=True, help='Model Structure')
parser.add_argument('--weight', type=str, default='', required=False, help='Model Weight')
parser.add_argument('--count', type=int, default=30, required=False, help='number of image to demo')
args = parser.parse_args()
if args.structure == '':
    args.structure = os.path.join(os.path.dirname(args.weight), 'structure.h5')

def read_img_and_resize(path, shape):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=shape[2])
    if shape[1] is None:
        img_shape = tf.shape(img)
        scale_factor = shape[0] / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
    else:
        img_width = shape[1]
    img = tf.image.resize(img, (shape[0], img_width))
    return img

# Specify GPU usuage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        for i in range(len(gpus)):
            mem = 1024 * 7 if i == 0 else 1024 * 9
            tf.config.set_visible_devices(gpus[i], 'GPU')
            tf.config.set_logical_device_configuration(gpus[i], [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Enable Jit to Accelerate
tf.config.optimizer.set_jit(True)

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

batch_size = config['batch_size_per_replica']
dataset_builder = DatasetBuilder(**config['dataset_builder'])

model_old = tf.keras.models.load_model(args.structure, compile=False, custom_objects={'BilinearInterpolation': BilinearInterpolation})
if args.weight: model_old.load_weights(args.weight)

input_tensor=model_old.input
output_tensor1=model_old.get_layer('ctc_logits').output
output_tensor2=model_old.get_layer('bilinear_interpolation').output
model = tf.keras.Model(inputs=input_tensor, outputs=[output_tensor1, output_tensor2])


model_pre = keras.layers.experimental.preprocessing.Rescaling(1./255)
model_post = CTCGreedyDecoder(config['dataset_builder']['table_path'])
model.build((None, 48, None, 3))
model.summary()

shutil.rmtree('demo', ignore_errors=True)
os.makedirs('demo', exist_ok=True)

img_paths = []
for prefix in ['*.jpg','*.png']:
    img_paths = img_paths + glob.glob(os.path.join(args.images, prefix))
    
for i,img_path in enumerate(img_paths):
    if i == args.count: break
    img_path = str(img_path)
    img = read_img_and_resize(img_path, config['dataset_builder']['img_shape'])
    img = tf.expand_dims(img, 0)
    padding = tf.zeros((1, tf.shape(img)[1], 50, 3))
    img = tf.concat([padding, img, padding], axis=2)

    result = model_pre(img)
    result, interpolate_img = model(result)
    result = model_post(result)
    
    print(f'Path: {img_path}, y_pred: {result[0].numpy()}',  f'probability: {result[1].numpy()}')

    predict_string=result[0].numpy()[0].decode('utf-8')
    embedding_string=f'{predict_string}'

    # Demonstrate Image
    img = img[0].numpy()
    img = img[..., ::-1].copy()
    img = cv2.rectangle(img, (0,0), (0 + 20, 10 ), (255,255,255), -1)
    img = cv2.putText(img, embedding_string, (0,10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    # STN Output Image
    stn_img = interpolate_img[0].numpy()
    h, w = stn_img.shape[:2]
    w = int(w * config['dataset_builder']['img_shape'][0] / h)
    h = int(config['dataset_builder']['img_shape'][0])
    stn_img = stn_img[..., ::-1]
    stn_img = cv2.resize(stn_img, (w,h))
    stn_img = (stn_img * 255.).astype(np.uint8)

    # Demo Image
    demo_img = np.hstack([img, stn_img])


    imgname = img_path.split('/')[-1]
    savepath=os.path.join('demo',imgname)
    cv2.imwrite(savepath, demo_img)