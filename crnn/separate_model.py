import argparse
from pathlib import Path
import cv2
import yaml
import tensorflow as tf
import os
import numpy as np
from layers.stn import BilinearInterpolation
import tensorflow_addons as tfa
print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, required=True, help='The config file path.')
args = parser.parse_args()

model = tf.keras.models.load_model(os.path.join(os.path.dirname(args.weight), 'structure.h5'), \
    compile=False, custom_objects={
        'BilinearInterpolation': BilinearInterpolation,
        'SpatialPyramidPooling2D': tfa.layers.SpatialPyramidPooling2D,
    })
model.load_weights(args.weight)
model.build((48,200,3))
model.summary()

# Get Crnn
input=x=tf.keras.Input((48,48,3))
construct=False
for layer in  model.layers:
    if 'conv2d_5' in layer.name: construct=True
    if construct: x = layer(x)
    if 'ctc_logits' in layer.name: construct=False
crnn = tf.keras.Model(input, x)
crnn.trainable=True
crnn.summary()
crnn_run = tf.function(lambda x: crnn(x))
crnn_concrete_func = crnn_run.get_concrete_function(tf.TensorSpec([1, 48, 48, 3], crnn.inputs[0].dtype))
crnn.save(os.path.join(os.path.dirname(args.weight), '..', 'crnn'), save_format="tf", signatures=crnn_concrete_func)
tf.saved_model.save(crnn, os.path.join(args.output_dir, 'pb_dir'), signatures=crnn_concrete_func)

# Get STN
input1=tf.keras.Input((48,200,3))
x1=input1
construct=False
for layer in  model.layers:
    if 'conv2d' == layer.name: construct=True
    print(layer.name)
    if construct: 
        if 'bilinear' in layer.name:
            print(input1)
            x1 = layer([input1, x1])
        else:
            x1 = layer(x1)
        print(layer.name, x1)
    if 'bilinear' in layer.name: construct=False

# input = model.get_layer('conv2d').input
# x = model.get_layer('bilinear_interpolation').output
# print('input', input)
# print('output', x)
stn = tf.keras.Model(input1, x1)
stn.trainable=True
stn.summary()
stn_run = tf.function(lambda x: stn(x))
stn_concrete_func = stn_run.get_concrete_function(tf.TensorSpec([1, 48, 200, 3], stn.inputs[0].dtype))
stn.save(os.path.join(os.path.dirname(args.weight), '..', 'stn'), save_format="tf", signatures=stn_concrete_func)
# tf.saved_model.save(stn, os.path.join(args.output_dir, 'pb_dir'), signatures=stn_concrete_func)



