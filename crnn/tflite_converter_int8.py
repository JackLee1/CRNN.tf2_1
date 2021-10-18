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

print(tf.__version__)

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
train_images=np.zeros((64, 48, 200, 3))
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]
# Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.weight, 'pb_dir'))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_enable_resource_variables = True
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
print('successfully convert tflite model')

# Save the model.
with open(os.path.join(args.export_dir, 'tflite_dir_int8','model.tflite'), 'wb') as f:
  f.write(tflite_model)
f.close()
print('successfully save tflite model')
#################################################################################################

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'tflite_dir_int8', 'model.tflite'))
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print('input_details', input_details)
print('output_details', output_details)
test_image = np.zeros((1,48,200,3))

interpreter.set_tensor(input_details["index"], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details["index"])[0]

print(output['output_0'].shape)
print('successfully inference in tflite model')