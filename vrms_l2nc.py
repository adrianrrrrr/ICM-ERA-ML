import csv
from glob import glob
from preprocessing_xgboost import XGBoostInputGenerator
from xgboost import XGBRegressor
import numpy as np
import time


def vrms_fnn_era5(prediction_corr, target_corr):
    N_target = len(target_corr)
    N_pred = len(prediction_corr)
    vrms_target = (np.sum(target_corr**2)/N_target)**0.5
    vrms_pred = (np.sum((target_corr-prediction_corr)**2)/N_pred)**0.5
    return vrms_target, vrms_pred, N_target

scat_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/test/"
model_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/saved_models/"
model_names = ["allvars_cpu_1000", "allvars_cpu_1500"]
#model_names = ["allvars_2000"]
vrms_stats_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l2_collocations/2020/vrms_stats/"
vrm_stats_pref = "2020_smart_"
vrm_stats_postf = "_test.csv"
model_ext = ".json"

'''
input_var_names = ['se_model_wind_divergence', 'vo', 'model_speed', 'msl',
                   'air_temperature', 'se_model_wind_curl', 'q', 'lon', 'sst',
                   'sst_dy', 'eastward_model_wind', 'uo', 'northward_model_wind',
                   'lat', 'model_dir']
'''
input_var_names = ['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'se_model_wind_curl', 'se_model_wind_divergence',
                   'msl', 'air_temperature', 'q', 'sst', 'sst_dx', 'sst_dy', 'uo', 'vo']
target_var_names = ['u_diff', 'v_diff']

for model_name in model_names:
    model_path = f"{model_dir}{model_name}{model_ext}"

    xgb_loaded = XGBRegressor(device='cuda')
    xgb_loaded.load_model(model_path)
    xgb_loaded.set_params(device='cuda')

    flist = glob(scat_dir + "*nc")
    flist.sort()
    vrms_stats = []
    vrms_stats_fn = f"{vrms_stats_dir}{vrm_stats_pref}{model_name}{vrm_stats_postf}"
    start = time.time()
    exec_times = []
    header = ["Filename", "ERA5", model_name, "N"]
    with open(vrms_stats_fn, mode='w', newline='') as vrms_file:
        vrms_writer = csv.writer(vrms_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vrms_writer.writerow(header)
        for a_fn in flist:
            xbg_vrms_generator = XGBoostInputGenerator(input_var_names, target_var_names, downsample_ratio=1)
            inputs, targets = xbg_vrms_generator.generate_dataset(a_fn)
            preds = xgb_loaded.predict(inputs)
            vrms_res = vrms_fnn_era5(preds, targets)
            row = [a_fn] + list(vrms_res)
            print(row)
            vrms_stats.append(row)
            vrms_writer.writerow(row)
        exec_times.append(time.time() - start)

for m_n, e_t in zip(model_names, exec_times):
    print(m_n, e_t, e_t/60)
