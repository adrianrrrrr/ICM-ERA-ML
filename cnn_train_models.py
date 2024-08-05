from ASCATA import *
from library_CNN import *
import time
#from xgboost import XGBRegressor
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from typing import Tuple, Dict, Any, List
from torchvision import datasets, transforms

import matplotlib

import matplotlib.pyplot as plt

import sklearn

start = time.time()

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cpu")

def generate_inputs_train(cnn_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix):

    # Full model load. uNETInputGenerator just load and create the dataset
    uNET_train_generator = uNETInputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                                output_dir=np_train_files_dir, downsample_ratio=1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    train_np_flist = uNET_train_generator.generate_np_files()
    uNET_eval_generator = uNETInputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                               output_dir=np_eval_files_dir, downsample_ratio=1,
                                               records_per_file=1e10,
                                               seed=123, out_file_prefix=prefix)
    eval_np_flist = uNET_eval_generator.generate_np_files()

    train_data = np.load(train_np_flist[0])
    eval_data = np.load(eval_np_flist[0])

    model_path = f"{save_model_folder}{prefix}1000.json"

    return model_path


# Agreed on last meeting
input_var_names = ['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']

scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                        'model':['eastward_model_wind', 'northward_model_wind']}
target_var_names = ['u_diff', 'v_diff']

# File path env setup. Currently local execution for Adrian's iMac
# Train period L3 01/01/2020 des
train_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_train/"
# Test period L3 01/02/2020 des
eval_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_test/"

np_train_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_np_data/train/"
np_eval_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_np_data/test/"

plots_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/plots/uNET_importance/"
save_model_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_uNET_models/"
file_prefix = "allvars_uNET_"


hparams = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'epochs': 10,
    }

if True:
    generate_inputs_train(hparams, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, 
    np_eval_files_dir, prefix=file_prefix)

train_fn = np_train_files_dir + "allvars_cpu_000.npz"
val_fn = np_eval_files_dir + "allvars_cpu_000.npz"

train_data = np.load(train_fn)
eval_data = np.load(val_fn)

# Actual uNET implementation


# Prepare the dataset and dataloader
X = np.float32(train_data['inputs']) # This to avoid errors when converting to Tensors
y = np.float32(train_data['targets'])
dataset = TensorDataset(torch.tensor(X).to(device), torch.tensor(y).to(device))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)

uNET = UNet(in_channels=12, out_channels=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(uNET.parameters(), lr=hparams['learning_rate'])

# Training loop
for epoch in range(hparams['epochs']):
    uNET.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = uNET(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    # Validation loop
    uNET.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = uNET(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("Training complete.")


'''
xgb.fit(train_data['inputs'], train_data['targets'], eval_set=[(eval_data['inputs'], eval_data['targets'])],
        verbose=True)
evals_metrics = xgb.evals_result()
#print(f"Estimators: {n_estimators}: {xgb.best_score}")
model_path = f"{save_model_folder}{file_prefix}{n_estimators}.json"
xgb.save_model(model_path)
print(input_var_names)
print("Training Time: %s seconds" % (str(time.time() - start)))

sort_indx = np.argsort(xgb.feature_importances_)

fig = plt.figure(figsize=(16, 6))
plt.barh(np.array(input_var_names)[sort_indx], xgb.feature_importances_[sort_indx])
plt.title(f"XGB {n_estimators} All RMSE {evals_metrics['validation_0']['rmse'][-1]:.4f}")
plt.savefig(f'{plots_folder}{file_prefix}{n_estimators}.png')
plt.show()
'''

