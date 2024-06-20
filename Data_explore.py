# -*- coding: utf-8 -*-
"""
This is a Python file containing the code for NWP bias prediction
Just examples

Author: Adrián Ramos González


"""
import time # We will have performance in mind since day 1

start_time = time.time()
# Dependencies
import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pygrib

from collections.abc import Iterable

elapsed_time = time.time()-start_time
print(elapsed_time)

def get_nwp_lat_lons(fpath: str):
    """
    Reads latitudes and longitudes from a Grib file.
    :param fpath: Path to grib file
    :return:
    Tuple of numpy arrays of latitudes and longitudes
    """
    grbs = pygrib.open(fpath)
    grb = grbs.read(1)[0]
    nwp_lats, nwp_lons = grb.latlons()
    return nwp_lats, nwp_lons

def get_nwp_var_ids(fpath: str, grib_key_var_name="cfVarName") -> dict:
    """
    Checks the variable names present in the grib file and returns their ids
    :param fpath: Path to the grib file
    :param grib_key_var_name: Key name for the short name of the Grib variables
    :return: Dict, {"variable_name": variable_id}
    example: {"u10n":1}
    """
    var_ids = {}
    with pygrib.open(fpath) as grbs:
        grbs.seek(0)
        for idx, grb in enumerate(grbs):
            var_ids[grb[grib_key_var_name]] = idx + 1
    return var_ids

def read_nwp_vars(fpath: str, var_namelist: Iterable):
    """
    Reads the values for the listed variables from the grib file
    :param fpath: Path to the grib file
    :param var_namelist: List with the variable names
    :return: list of numpy arrays or numpy masked arrays with the values
    """
    nwp_vars_data = []
    with pygrib.open(fpath) as grbs:
        for var_name in var_namelist:
            selected_grbs = grbs.select(cfVarName=var_name)
            for grb in selected_grbs:
                nwp_vars_data.append(grb.values)
    return nwp_vars_data

def read_nwp_var(fpath, var_id):
    """
    Reads variable from grib file by its id
    :param fpath: Path to the grib file
    :param var_id: Variable id in the grib file
    :return: variable in numpy array
    """
    with pygrib.open(fpath) as grbs:
        grb = grbs.select()[var_id]
        nwp_var = grb.values
    return nwp_var

File = "ERA5/nwp_2020032_06_00.grib"

ids = get_nwp_var_ids(File)
#X_data = read_nwp_var("ERA5/nwp_2020032_06_00.grib", ids['sst']-1)
X = read_nwp_vars("ERA5/nwp_2020032_06_00.grib", ids)

X = X[0].tolist(-1)
