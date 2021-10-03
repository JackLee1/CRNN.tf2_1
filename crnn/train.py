import argparse
import pprint
import shutil
import os
from pathlib import Path
import tensorflow as tf
import yaml
from dataset_factory import DatasetBuilder
from losses import CTCLoss, LossBox
from metrics import SequenceAccuracy
from models import build_model
import json
from callbacks.callbacks import ImageCallback

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True, help='The path to save the models, logs, etc.')
parser.add_argument('--weight', type=str, default='', required=False, help='The pretrained weight of model.')
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--point4', dest='point4', action='store_true')
feature_parser.add_argument('--no-point4', dest='point4', action='store_false')
parser.set_defaults(point4=True)
args = parser.parse_args()
os.makedirs(f'{args.save_dir}/weights', exist_ok=True)
os.makedirs(f'{args.save_dir}/configs', exist_ok=True)
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
tf.config.optimizer.set_jit(True)
# Load and Save Config
with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
for filename in ['models.py', 'dataset_factory.py', 'losses.py']:
    shutil.copyfile(f'./crnn/{filename}', f'{args.save_dir}/configs/{filename}')
shutil.copyfile(args.config, f'{args.save_dir}/configs/config.yml')
pprint.pprint(config)



args.save_dir.mkdir(exist_ok=True)
shutil.copy(args.config, args.save_dir / args.config.name)

strategy = tf.distribute.MirroredStrategy()
batch_size = config['batch_size_per_replica'] * strategy.num_replicas_in_sync
print(config['dataset_builder'])
dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=args.point4)
train_ds = dataset_builder(config['train_ann_paths'], batch_size, True)
val_ds = dataset_builder(config['val_ann_paths'], batch_size, False)

with strategy.scope():
    model, stn_model = build_model(dataset_builder.num_classes,
                        require_coords=args.point4,
                        weight=args.weight,
                        img_shape=config['dataset_builder']['img_shape'])
    lr=config['lr_schedule']['initial_learning_rate']
    opt=tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss=[[CTCLoss()],[LossBox()]], metrics={
        "ctc_logits":SequenceAccuracy()
    })
    model.save(os.path.join(args.save_dir, 'weights', 'structure.h5'),include_optimizer=False)
    model.summary()
# Use validation accuracy to make sure load the right model
if args.weight:
    model.evaludate(val_ds)

best_model_prefix = 'best_model'
best_model_path = f'{args.save_dir}/weights/{best_model_prefix}.h5'
model_prefix = '{epoch}_{val_loss:.4f}_{val_ctc_logits_sequence_accuracy:.4f}' if args.point4 else '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
model_path = f'{args.save_dir}/weights/{model_prefix}.h5'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, period=10),
    tf.keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs', **config['tensorboard']),
    ImageCallback(f'{args.save_dir}/images/', train_ds, stn_model, require_coords=args.point4),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.318, patience=15, min_lr=1e-8, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=51),
]
model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds)
