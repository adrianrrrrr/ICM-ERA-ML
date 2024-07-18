from preprocessing_xgboost import XGBoostInputGenerator
import time
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

def generate_inputs_train(xgboost_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix):
    xgb = XGBRegressor(**xgboost_params)

    # Full model
    xbg_train_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                                output_dir=np_train_files_dir, downsample_ratio=0.1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    train_np_flist = xbg_train_generator.generate_np_files()
    xbg_eval_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                               output_dir=np_eval_files_dir, downsample_ratio=0.1,
                                               records_per_file=1e10,
                                               seed=123, out_file_prefix=prefix)
    eval_np_flist = xbg_eval_generator.generate_np_files()
    train_data = np.load(train_np_flist[0])
    eval_data = np.load(eval_np_flist[0])

    xgb.fit(train_data['inputs'], train_data['targets'], eval_set=[(eval_data['inputs'], eval_data['targets'])],
            verbose=True)
    evals_metrics = xgb.evals_result()
    model_path = f"{save_model_folder}{prefix}1000.json"
    xgb.save_model(model_path)
    return xgb, evals_metrics, model_path

# Performance optimizacion for XGBoost executed in iMac 2020 AMD Radeon Pro 5300 4 GB to prevent CPU usage
# It do not work without an AMD Accelerated Cloud (AAC) account. Unable to make one without invitation
'''
git clone https://github.com/ROCmSoftwarePlatform/xgboost.git
cd xgboost
git submodule update --init --recursive

mkdir build
cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a" -DUSE_RCCL=1 ../
make -j

cd python-package
pip install .

'''
input_var_names = ['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'se_model_wind_curl', 'se_model_wind_divergence',
                   'msl', 'air_temperature', 'q', 'sst', 'sst_dx', 'sst_dy', 'uo', 'vo']

scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                        'model':['eastward_model_wind', 'northward_model_wind']}
target_var_names = ['u_diff', 'v_diff']

# File path env setup. Currently local execution for Adrian's iMac
# Train period from 02/01/2020 - 06/03/2020 both included
train_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/train/"
# Test period from  10/03/2020 - 01/05/2020 both included
eval_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/test/"

np_train_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/xgb_np_data/train/"
np_eval_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/xgb_np_data/test/"

plots_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/plots/xgboost_importance/"
save_model_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/saved_models/"
file_prefix = "allvars_cpu_"

n_estimators = 1500
xgboost_params = {
                'early_stopping_rounds':  False,
                'booster': 'gbtree',
                'device': 'cpu', # Change it for cloud running in CALCULA services. Adrian's iMac has a Radeon non-supported GPU
                'tree_method': 'hist',
                #'tree_method': 'gpu_hist',
                'eval_metric': 'rmse',
                'n_estimators': n_estimators,
                'min_child_weight': 1,
                'gamma': 0.2,
                'subsample': .5,
                'colsample_bytree': 0.5,
                'max_depth': 10,
                'alpha': 0.2,
                'learning_rate': 0.01,
                'objective': 'reg:squarederror',
                'verbosity': 2
                }



# Generating the input data with the helper function generate_inputs_train
if False:
        generate_inputs_train(xgboost_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, 
                        np_eval_files_dir, prefix=file_prefix)





xgb = XGBRegressor(**xgboost_params)
train_fn = np_train_files_dir + "allvars_cpu_000.npz"
val_fn = np_eval_files_dir + "allvars_cpu_000.npz"

train_data = np.load(train_fn)
eval_data = np.load(val_fn)

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
xbg_train_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                            output_dir=np_train_files_dir, downsample_ratio=0.1,
                                            records_per_file=1e10,
                                            seed=123, out_file_prefix=file_prefix)
train_np_flist = xbg_train_generator.generate_np_files()
xbg_eval_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                           output_dir=np_eval_files_dir, downsample_ratio=0.1,
                                           records_per_file=1e10,
                                           seed=123, out_file_prefix=file_prefix)
eval_np_flist = xbg_eval_generator.generate_np_files()
start = time.time()
train_data = np.load(train_np_flist[0])
eval_data = np.load(eval_np_flist[0])

estimators = [1500] #Number of iterations 
# Train 
for n_estimators in estimators:
    start = time.time()
    xgboost_params = {
            'early_stopping_rounds':  False,
            'booster': 'gbtree',
            'device': 'cuda',
            #'tree_method': 'hist',
            'tree_method': 'gpu_hist',
            'eval_metric': 'rmse',
            'n_estimators': n_estimators,
            'min_child_weight': 1,
            'gamma': 0.2,
            'subsample': .5,
            'colsample_bytree': 0.5,
            'max_depth': 10,
            'alpha': 0.2,
            'learning_rate': 0.01,
            'objective': 'reg:squarederror',
            'verbosity': 2
            }

    xgb = XGBRegressor(**xgboost_params)
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

# Generation of inputs
[xgb, evals_metrics,model_path] = generate_inputs_train(xgboost_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix)
 
'''