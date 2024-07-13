import argparse
import os
import random
import logging.config

import numpy as np
import tensorflow as tf
# TensorFlow Addons has stopped development. Minimal maintenance releases until May 2024.
import tensorflow_addons as tfa

from prediction.my_image_model.image_model import architecture
from prediction.my_image_model.training_process import train
from prediction.my_image_model.signal_preprocess import load_singals_datasets, gen_images_dataset
import prediction.my_image_model.model_params as params
import prediction.model.model_util as util

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('training')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
# fix random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.compat.v1.set_random_seed(SEED)
tf.random.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, default=params.AUDIO_TRAIN_NAME, help="Name of this programe.")
parser.add_argument("--task", type=str, default=params.TASK, help="Name of the task.")
parser.add_argument("--data_name", type=str, default=params.DATA_NAME, help="Original data path.")
parser.add_argument("--is_aug", type=bool, default=False, help="Add data augmentation.")
# parser.add_argument("--restore_if_possible", type=bool, default=False, help="Restore variables.")
parser.add_argument("--epoch", type=int, default=params.NUM_EPOCHS, help="Maximum epoch to train.")
parser.add_argument("--test_epoch", type=int, default=None, help="Maximum epoch to train.")
parser.add_argument("--early_stop", type=str, default=params.EARLY_STOP,
                    help="The indicator on validation set to stop training.")
parser.add_argument("--is_diff", type=bool, default=True, help="Whether to use differential learing rate.")
# parser.add_argument("--enable_checkpoint", type=bool, default=True, help="Use pretrained base model.")
# parser.add_argument("--train_base", type=bool, default=True, help="Fine-tuning base model")

parser.add_argument("--reg_l2", type=float, default=params.L2, help="L2 regulation.")
parser.add_argument("--lr_decay", type=float, default=params.LEARNING_RATE_DECAY, help="Learning rate decay rate.")
parser.add_argument("--dropout_rate", type=float, default=params.DROPOUT_RATE, help="Dropout rate.")
parser.add_argument("--trained_layers", type=int, default=params.TRAINED_LAYERS,
                    help="The number of base model layers to be fine tuned.")
parser.add_argument("--num_units", type=int, default=params.NUM_UNITS, help="The numer of unit in network.")
parser.add_argument("--lr_base", type=float, default=params.LR_BASE, help="Learning rate for base model layers.")
parser.add_argument("--lr_top", type=float, default=params.LR_TOP, help="Learning rate for top layers.")
parser.add_argument("--model_type", type=str, default=params.MODEL_TYPE, help="Base model type.")

FLAGS, _ = parser.parse_known_args()
FLAGS = vars(FLAGS)

# Paths
# tensorboard_dir = os.path.join(params.TENSORBOARD_DIR,
#                                f"{FLAGS['train_name']}_{FLAGS['model_type']}")  # ./data/tensorbord/
audio_ckpt_dir = os.path.join(params.AUDIO_CHECKPOINT_DIR,
                              f"{FLAGS['train_name']}_{FLAGS['model_type']}")  # ./data/train/
name_pre = f"Drp{FLAGS['dropout_rate']}_Uni{FLAGS['num_units']}_L2{FLAGS['reg_l2']}_numL{FLAGS['trained_layers']}"
name_mid = f"DC{FLAGS['lr_decay']}_LR{FLAGS['lr_base']}_{FLAGS['lr_top']}"
name_pos = f"Aug{FLAGS['is_aug']}"
name_all = f"{name_pre}__{name_mid}__{name_pos}__"
logger.info(f"model: {FLAGS['model_type']}\nsave: {name_all}")
logfile = os.path.join(str(audio_ckpt_dir), f"{name_all}_log.txt")
checkpoint_path = os.path.join(str(audio_ckpt_dir), f"{name_all}{params.AUDIO_CHECKPOINT_NAME}")
util.maybe_create_directory(audio_ckpt_dir)

# Parameters
lr_top = FLAGS['lr_top']  # Initial learning rate for classification layers
lr_base = FLAGS['lr_base']  # Initial learning rate for fine-tuning base model
decay_rate = FLAGS['lr_decay']
decay_steps = 30000
reg_l2 = FLAGS['reg_l2']
dropout = FLAGS['dropout_rate']
num_finetune_layers = FLAGS['trained_layers']  # Number of top layers to fine-tune
num_units = FLAGS['num_units']  # Number of units in the additional dense layer
max_epochs = FLAGS['epoch']
test_epoch = FLAGS['test_epoch']
before_finetune_epochs = 5
early_stop_patience = params.PATIENCE
base_model_type = FLAGS['model_type']

# Additional parameters
if base_model_type == 'resnet':
    base_model_t = tf.keras.applications.resnet.ResNet50
    preprocess_input = tf.keras.applications.resnet.preprocess_input  # caffe (RGB -> BRG, zero-center)
    num_after_pooling_units = 2048
elif base_model_type == 'densenet':
    base_model_t = tf.keras.applications.densenet.DenseNet121
    preprocess_input = tf.keras.applications.densenet.preprocess_input  # torch ([0;1], normalisation)
    num_after_pooling_units = 1024
else:
    raise Exception("Invalid model")

# Prepare the datasets signals
train_set_signals, val_set_signals, test_set_signals = load_singals_datasets(samples_count=None, ratios=(0.7, 0.2, 0.1))
train_dataset = gen_images_dataset(train_set_signals, preprocess=preprocess_input, shape=(224, 224))
val_dataset = gen_images_dataset(val_set_signals, preprocess=preprocess_input, shape=(224, 224))
test_dataset = gen_images_dataset(test_set_signals, preprocess=preprocess_input, shape=(224, 224))
# dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

base_model, model = architecture(base_model_trained=base_model_t,
                                 num_units=num_units, num_after_pooling_units=num_after_pooling_units,
                                 reg_l2=reg_l2, dropout=dropout, base_model_trainable=False)
model.summary(show_trainable=True)

# Create optimizer for the fine-tuning stage
lr_base_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr_base,
    decay_steps=decay_steps,
    decay_rate=decay_rate)
lr_top_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr_top,
    decay_steps=decay_steps,
    decay_rate=decay_rate)
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=lr_base_schedule),
    tf.keras.optimizers.Adam(learning_rate=lr_top_schedule)
]
optimizers_and_layers = [(optimizers[0], base_model), (optimizers[1], model.layers[2:])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_tpr_tnr', patience=early_stop_patience, mode='max',
                                                  verbose=1, restore_best_weights=True)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy',
             'categorical_crossentropy']
)

# logger.info(model.compiled_metrics._metrics)
# Train the new last layers for few epochs - for the model to converge on the new data
logger.info("Train the new last layers")
logger.info(f'Model layers: {len(model.layers)}')
logger.info(f'Base model layers: {len(base_model.layers)}')
logger.info(f'Model trainable variables - before fine-tune: {len(model.trainable_variables)}')
logger.info(f'Base model trainable variables - before fine-tune: {len(base_model.trainable_variables)}')
train(model, train_dataset, val_dataset, test_data=test_dataset, epochs=before_finetune_epochs, metrics_file=logfile,
      save_checkpoint_file=checkpoint_path)

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
fine_tune_at = len(base_model.layers) - num_finetune_layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
logger.info(f'Model trainable variables - at fine-tune: {len(model.trainable_variables)}')
logger.info(f'Base model trainable variables - at fine-tune: {len(base_model.trainable_variables)}')

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)
model.summary(show_trainable=True)

logger.info("Fitting the end-to-end model")

train(model, train_dataset, val_dataset, test_data=test_dataset, epochs=max_epochs, start_epoch=before_finetune_epochs,
      metrics_file=logfile, early_stopping=False, test_epoch=test_epoch, save_checkpoint_file=checkpoint_path)

logger.info("END OF TRAINING")
