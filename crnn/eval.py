import argparse
import pprint
import tensorflow as tf
import yaml

from dataset_factory import DatasetBuilder
from losses import CTCLoss, LossBox
from metrics import SequenceAccuracy, EditDistance
from models import build_model
from layers.stn import BilinearInterpolation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='The config file path.')
parser.add_argument('--weight', type=str, default='', required=False, help='The saved weight path.')
parser.add_argument('--structure', type=str, required=True, help='The saved structure path.')
parser.add_argument('--point4', type=bool, default=False, required=False)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)['eval']
pprint.pprint(config)

dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=args.point4)
ds = dataset_builder(config['ann_paths'], config['batch_size'], False)
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
model.evaluate(ds)