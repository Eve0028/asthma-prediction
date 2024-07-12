# Define VGGish parameters
import os

DATA_NAME = "audio_0124"
TF_DATA_DIR = ""
EMBEDDING_SIZE = 128
sample_width = 96
sample_height = 64

# Paths
signals_data_dir = os.path.join(TF_DATA_DIR, DATA_NAME)  # ./data
SAMPLES_COUNT = 20

# Customizable parameters
dropout_rate1 = 0.5
fc_neurons = 512
reg_l2 = 0.01
dropout_rate2 = 0.5
early_stop_patience = 3
number_of_finetune_layers = 5
max_epochs = 20
initial_epochs = 10
sample_rate = 16000
batch_size = 1
