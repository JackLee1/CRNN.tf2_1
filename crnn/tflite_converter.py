import argparse
from pathlib import Path
import cv2
import yaml
import tensorflow as tf
import os
import numpy as np
from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from layers.stn import BilinearInterpolation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--pre', type=str, help='pre processing.')
parser.add_argument('--weight',type=str,required=True,help='directory of weight')
parser.add_argument('--export_dir',type=str,default='export_dir', help='Model output dir')
args = parser.parse_args()
args.output_dir='export_dir'
os.makedirs(args.output_dir, exist_ok=True)
for subdir in ['pb_dir', 'tflite_dir']:
    os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)


with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

with open(config['table_path']) as f:
    num_classes = len(f.readlines())

# Testing Load and Inference pb graph
loaded=tf.saved_model.load(os.path.join(args.weight, 'pb_dir'))
print(list(loaded.signatures.keys()))
infer = loaded.signatures[list(loaded.signatures.keys())[0]]
print(infer.structured_outputs)

# pb Inference
image = np.zeros((48,200,3))
print(image.shape)
image = (np.expand_dims(image, axis=0) / 255.0).astype(np.float32) # (1,14,63,3)
result = loaded(image)
print("successfully do inference in savemodel")
print(np.argmax(result, axis=-1))
print("CRNN has {} trainable variables, ...".format(len(loaded.trainable_variables)))
for v in loaded.trainable_variables:
    print(v.name)
#################################################################################################


# Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.weight, 'pb_dir'))
tflite_model = converter.convert()
print('successfully convert tflite model')

# Save the model.
with open(os.path.join(args.export_dir, 'tflite_dir','model.tflite'), 'wb') as f:
  f.write(tflite_model)
f.close()
print('successfully save tflite model')
#################################################################################################

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'tflite_dir','model.tflite'))
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,48,200,3), dtype=tf.float32))
print(output.keys())
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['output_0'].shape)
print('successfully inference in tflite model')