import argparse
from pathlib import Path
import cv2
import yaml
import tensorflow as tf
import os
import numpy as np

print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--pre', type=str, help='pre processing.')
parser.add_argument('--weight',type=str, required=True, help='directory of weight')
parser.add_argument('--export_dir',type=str, required=True, help='directory of export dir')
parser.add_argument('--fp32', default=False, action="store_true")
parser.add_argument('--fp16', default=False, action="store_true")
parser.add_argument("--int8", default=False, action="store_true")

args = parser.parse_args()
os.makedirs(args.export_dir, exist_ok=True)

dir_list=['pb_dir']
if args.fp32: dir_list.append('tflite_fp32')
if args.fp16: dir_list.append('tflite_fp16')
if args.int8: dir_list.append('tflite_int8')

for subdir in dir_list:
    os.makedirs(os.path.join(args.export_dir, subdir), exist_ok=True)

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

with open(config['table_path']) as f:
    num_classes = len(f.readlines())

# Testing Load and Inference pb graph
loaded=tf.saved_model.load(args.weight)
print(list(loaded.signatures.keys()))
infer = loaded.signatures[list(loaded.signatures.keys())[0]]
print('infer.structured_outputs', infer.structured_outputs)

# pb Inference
image = np.zeros((48,200,3))
image = (np.expand_dims(image, axis=0) / 255.0).astype(np.float32) # (1,14,63,3)
print('input image detaial', image.shape, image.dtype)
result = loaded(image)
print("successfully do inference in savemodel")
print('result of output', result.shape, result.dtype)
print("CRNN has {} trainable variables, ...".format(len(loaded.trainable_variables)))
# for v in loaded.trainable_variables:
#     print(v.name)

if args.fp32:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    tflite_model = converter.convert()
    print('successfully convert tflite fp32 model')

    # Save the model.
    with open(os.path.join(args.export_dir, 'tflite_fp32','model.tflite'), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite fp32 model')
    
    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'tflite_fp32','model.tflite'))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('fp32 input_details', input_details)
    print('fp32 output_details', output_details)

    my_signature = interpreter.get_signature_runner()
    # my_signature is callable with input as arguments.
    output = my_signature(x=tf.constant([1.0], shape=(1,48,200,3), dtype=tf.float32))
    print(output.keys())
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    print(output['output_0'].shape, output['output_0'].dtype)
    print('successfully inference in tflite fp32 model')

if args.fp16:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    print('successfully convert tflite fp16 model')

    # Save the model.
    with open(os.path.join(args.export_dir, 'tflite_fp16','model.tflite'), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite fp16 model')
    
    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'tflite_fp16','model.tflite'))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('fp16 input_details', input_details)
    print('fp16 output_details', output_details)


    my_signature = interpreter.get_signature_runner()
    # my_signature is callable with input as arguments.
    output = my_signature(x=tf.constant([1.0], shape=(1,48,200,3), dtype=tf.float32))
    print(output.keys())
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    print(output['output_0'].shape)
    print('successfully inference in tflite fp16 model')

if args.int8:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    train_images=np.zeros((200, 48, 200, 3), dtype=np.float32)
    def representative_data_gen():
      for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]
      
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_enable_resource_variables = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
      tf.lite.OpsSet.SELECT_TF_OPS,
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    print('successfully convert tflite int8 model')

    # Save the model
    with open(os.path.join(args.export_dir, 'tflite_int8','model.tflite'), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite int8 model')


    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'tflite_int8', 'model.tflite'))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('int8 input_details', input_details)
    print('int8 output_details', output_details)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], np.zeros((1,48,200,3), dtype=np.int8))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    print(output.shape)
    print('successfully inference in tflite int8 model')


