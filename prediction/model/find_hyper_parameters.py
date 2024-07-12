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

# lr1 = [1e-5, 1e-4]
# lr2 = [1e-4, 1e-3]
# lr_base = [1e-6, 1e-4]
# reg_l2 = [1e-7, 1e-3]
# dropout_rate = [0.5, 0.7]  # 1 - dropout_rate ??
# trained_layers = [8, 12]
# lr_decay = [0.8]
# num_units = [64]

# combinations = [{'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.7, 'num_units': 64,
#                  'trained_layers': 10},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.7, 'num_units': 128,
#                  'trained_layers': 10},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.7, 'num_units': 128,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.7, 'num_units': 64,
#                  'trained_layers': 12}]

# combinations = [{'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.5, 'num_units': 128,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-07, 'dropout_rate': 0.5, 'num_units': 64,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.5, 'num_units': 64,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.6, 'num_units': 64,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.6, 'num_units': 64,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-4, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.6, 'num_units': 40,
#                  'trained_layers': 12}]

# combinations = [{'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.5, 'num_units': 30,
#                  'trained_layers': 12},
#                 {'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-06, 'dropout_rate': 0.5, 'num_units': 30,
#                  'trained_layers': 8},
#                 {'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-05, 'dropout_rate': 0.5, 'num_units': 30,
#                  'trained_layers': 10},
#                 {'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-04, 'dropout_rate': 0.5, 'num_units': 30,
#                  'trained_layers': 10}]

combinations = [{'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-03, 'dropout_rate': 0.5, 'num_units': 30,
                 'trained_layers': 12},
                {'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-03, 'dropout_rate': 0.5, 'num_units': 20,
                 'trained_layers': 12},  # Units
                {'lr1': 1e-05, 'lr2': 1e-04, 'lr_decay': 0.8, 'reg_l2': 1e-03, 'dropout_rate': 0.5, 'num_units': 20,
                 'trained_layers': 8}]  # Layers

for comb in combinations:
    command = ["python", "model_train_vad.py",
               f"--lr1={comb['lr1']}", f"--lr2={comb['lr2']}",
               f"--lr_decay={comb['lr_decay']}", f"--reg_l2={comb['reg_l2']}",
               f"--dropout_rate={comb['dropout_rate']}",
               f"--num_units={comb['num_units']}",
               f"--trained_layers={comb['trained_layers']}"]
    subprocess.run(command)

# param_combinations = itertools.product(lr_base, lr_decay, reg_l2, dropout_rate, num_units, trained_layers)

# for params in param_combinations:
#     command = ["python", "model_train_vad.py",
#                f"--lr1={params[0]}", f"--lr2={params[0] * 10:e}",
#                f"--lr_decay={params[1]}", f"--reg_l2={params[2]}",
#                f"--dropout_rate={params[3]}",
#                f"--num_units={params[4]}",
#                f"--trained_layers={params[5]}"]
#     subprocess.run(command)
