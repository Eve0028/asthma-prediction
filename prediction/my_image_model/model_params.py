"""
Global parameters for the model.

"""

from os.path import join as pjoin

# Training
TASK = "main"  # pre/main
AUDIO_TRAIN_NAME = "asthma_model"  # train name
MODEL_TYPE = 'densenet'
NUM_EPOCHS = 100
BATCH_SIZE = 1
L2 = 1e-6
PATIENCE = 20
DROPOUT_RATE = 0.5
LR_BASE = 0.0001
LR_TOP = 0.001
LEARNING_RATE_DECAY = 0.98
SR_VGG = 16000
EARLY_STOP = "AUC"
SAMPLES_COUNT = 1364  # 4774 # 1364 - without symptoms; all samples (1:1 - classes proportion)

# Architectural constants.
EMBEDDING_SIZE = 128  # Size of embedding layer.
NUM_CLASSES = 2
TRAINED_LAYERS = 20  # all layers: ResNet50: 175, DenseNet121: 427

INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-5  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.
NUM_UNITS = 128  # hidden units
# PREVALANCE = 0.5  # the percentage of covid in the population

ISCOMB = False
ISAUG = False

# Data
TF_DATA_DIR = "../data/"
# DATA_NAME = "data/audio_0124"
DATA_NAME = "audio_0124"
IS_ADDDATA = False

# Checkpoint,no need to change
TENSORBOARD_DIR = "../data/tensorboard"  # Tensorboard
AUDIO_CHECKPOINT_DIR = "../data/train"
AUDIO_CHECKPOINT_NAME = "mymodel"

# ResNet
RESNET_CHECKPOINT_DIR = "../resnet"
RESNET_CHECKPOINT_NAME = "resnet_model.ckpt"
RESNET_PCA_PARAMS_NAME = "resnet_pca_params.npz"
RESNET_CHECKPOINT = pjoin(RESNET_CHECKPOINT_DIR, RESNET_CHECKPOINT_NAME)
RESNET_PCA_PARAMS = pjoin(RESNET_CHECKPOINT_DIR, RESNET_PCA_PARAMS_NAME)

# RESNET_INPUT_OP_NAME = "resnet/input_features"
# RESNET_INPUT_TENSOR_NAME = RESNET_INPUT_OP_NAME + ":0"
# RESNET_OUTPUT_OP_NAME = "resnet/embedding"
# RESNET_OUTPUT_TENSOR_NAME = RESNET_OUTPUT_OP_NAME + ":0"
