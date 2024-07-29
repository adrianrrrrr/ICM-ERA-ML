from ASCATA import InputGenerator
from library_CNN import CNNRegressor
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

def generate_inputs_train(cnn_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix):
    cnn = CNNRegressor(**cnn_params)

    # Full model
    cnn_train_generator = InputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                                output_dir=np_train_files_dir, downsample_ratio=0.1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    train_np_flist = cnn_train_generator.generate_np_files()
    xbg_eval_generator = InputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                               output_dir=np_eval_files_dir, downsample_ratio=0.1,
                                               records_per_file=1e10,
                                               seed=123, out_file_prefix=prefix)
    eval_np_flist = xbg_eval_generator.generate_np_files()
    train_data = np.load(train_np_flist[0])
    eval_data = np.load(eval_np_flist[0])

    cnn.fit(train_data['inputs'], train_data['targets'], eval_set=[(eval_data['inputs'], eval_data['targets'])],
            verbose=True)
    evals_metrics = cnn.evals_result()
    model_path = f"{save_model_folder}{prefix}1000.json"
    cnn.save_model(model_path)
    return cnn, evals_metrics, model_path


input_var_names = ['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'se_model_wind_curl', 'se_model_wind_divergence',
                   'msl', 'air_temperature', 'q', 'sst', 'sst_dx', 'sst_dy', 'uo', 'vo']

scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                        'model':['eastward_model_wind', 'northward_model_wind']}
target_var_names = ['u_diff', 'v_diff']

# File path env setup. Currently local execution for Adrian's iMac
# Train period from 02/01/2020 - 06/03/2020 both included
train_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/cnn_train/"
# Test period from  10/03/2020 - 01/05/2020 both included
eval_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/cnn_test/"

np_train_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/cnn_np_data/train/"
np_eval_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/cnn_np_data/test/"

plots_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/plots/cnn_importance/"
save_model_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/saved_cnn_models/"
file_prefix = "allvars_cnn_"


cnn_params = {
'lon',
'lat',
'eastward_model_wind',
'northward_model_wind',
'model_speed',
'model_dir',
'se_model_wind_curl',
'se_model_wind_divergence',
'msl',
'air_temperature',
'q',
'sst',
'sst_dx',
'sst_dy',
'uo',
'vo'
}

