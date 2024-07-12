import itertools
import subprocess

# lr1 = [1e-5, 1e-4, 1e-3]
# lr2 = [1e-4, 1e-3, 1e-2]
# lr_decay = [0.8, 0.9, 0.99]
# reg_l2 = [1e-7, 1e-5, 1e-3]
# dropout_rate = [0.3, 0.5, 0.7]
# num_units = [32, 64, 128, 256]
# trained_layers = [2, 6, 10]

# lr1 = [1e-5, 1e-3]
# lr2 = [1e-4, 1e-2]
# lr_decay = [0.8, 0.99]
# reg_l2 = [1e-7, 1e-3]
# dropout_rate = [0.3, 0.7]
# num_units = [32, 64, 128]
# trained_layers = [2, 10]

# lr1 = [1e-5, 1e-3]
# lr2 = [1e-4, 1e-2]
# reg_l2 = [1e-7, 1e-3]
# dropout_rate = [0.3, 0.7]
# trained_layers = [2, 10]
# lr_decay = [0.8]
# num_units = [64]

lr_base = [1e-6, 1e-4]
# lr1 = [1e-5, 1e-4]
# lr2 = [1e-4, 1e-3]
reg_l2 = [1e-5, 1e-3]
dropout_rate = [0.3, 0.5]
trained_layers = [16, 32]
lr_decay = [0.8]
num_units = [40, 64, 128]
model_type = ['resnet', 'densenet']
# last layer units: ResNet50: 2048, DenseNet121: 1024
# all layers: ResNet50: 175, DenseNet121: 427

# combinations = [
#     {'lr_base': 1e-04, 'lr2': 1e-03, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 1024,
#      'trained_layers': 12, 'model_type': 'resnet'},
#     {'lr_base': 1e-04, 'lr2': 1e-03, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 512,
#      'trained_layers': 12, 'model_type': 'resnet'},  # Best
#     {'lr_base': 1e-04, 'lr2': 1e-03, 'lr_decay': 0.8, 'reg_l2': 1e-05, 'dropout_rate': 0.3, 'num_units': 512,
#      'trained_layers': 12, 'model_type': 'densenet'},
#     {'lr_base': 1e-04, 'lr2': 1e-03, 'lr_decay': 0.8, 'reg_l2': 1e-04, 'dropout_rate': 0.3, 'num_units': 206,
#      'trained_layers': 12, 'model_type': 'densenet'}]

# combinations = [
#     {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 512,
#      'trained_layers': 22, 'model_type': 'resnet'},  # lr
#     {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.2, 'num_units': 512,
#      'trained_layers': 22, 'model_type': 'resnet'},  # lr + dropout
#     {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-05, 'dropout_rate': 0.3, 'num_units': 512,
#      'trained_layers': 12, 'model_type': 'densenet'},  # lr
#     {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 512,
#      'trained_layers': 12, 'model_type': 'densenet'},  # lr + l2
#     {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 206,
#      'trained_layers': 12, 'model_type': 'densenet'}]  # units

combinations = [
    {'lr_base': 1e-04, 'lr_top': 1e-03, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.3, 'num_units': 512,
     'trained_layers': 12, 'model_type': 'resnet', 'test_epoch': 8, 'max_epoch': 24},
    {'lr_base': 1e-05, 'lr_top': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-05, 'dropout_rate': 0.3, 'num_units': 512,
     'trained_layers': 12, 'model_type': 'densenet', 'test_epoch': 38, 'max_epoch': 54}]

for comb in combinations:
    command = ["python", "train_pipe.py",
               f"--lr_base={comb['lr_base']}", f"--lr_top={comb['lr_top']}",
               f"--lr_decay={comb['lr_decay']}", f"--reg_l2={comb['reg_l2']}",
               f"--dropout_rate={comb['dropout_rate']}",
               f"--num_units={comb['num_units']}",
               f"--trained_layers={comb['trained_layers']}",
               f"--model_type={comb['model_type']}",
               f"--epoch={comb['max_epoch']}",
               f"--test_epoch={comb['test_epoch']}"]
    subprocess.run(command)

# param_combinations = itertools.product(lr_base, lr_decay, reg_l2, dropout_rate, num_units, trained_layers, model_type)
#
# for params in param_combinations:
#     command = ["python", "train_pipe.py",
#                f"--lr1={params[0]}", f"--lr2={params[0]*10:e}",
#                f"--lr_decay={params[1]}", f"--reg_l2={params[2]}",
#                f"--dropout_rate={params[3]}",
#                f"--num_units={params[4]}",
#                f"--trained_layers={params[5]}",
#                f"--model_type={params[6]}"]
#     subprocess.run(command)
