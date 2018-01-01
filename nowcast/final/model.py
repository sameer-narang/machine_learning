#!/usr/bin/python3.5

import calendar
import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
import sklearn

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import features
import raw_data
import util

from features import *
from raw_data import *
from util import *

y_hat_frbny = None

RNDM_SEED_1 = 4
RNDM_SEED_2 = 11
TEST_SIZE = 0.10
NA_FILL_VAL=None

MTHS_TO_COMBINE=3

class Model (object):
    # sn: default is quarterly bucketing of data series to reduce noise that more frequent data usually has
    def __init__ (self, days_prior=0, seed=RNDM_SEED_1):
        self._days_prior = days_prior
        self._seed = seed

        self._X = None
        self._y = None
        self._model = None
        self._y_scaler = None


    def prepare_training_data (self, x_df, y_df, scale=True):
        X = np.delete (x_df.drop (columns=['Date'], axis=1).as_matrix (), 0, axis=1)
        y = np.delete (y_df.drop (columns=['Date'], axis=1).as_matrix (), 0, axis=1)

        self._X_train_and_val, self._X_test, self._y_train_and_val, self._y_test = \
            train_test_split (X, y, test_size=TEST_SIZE, random_state=self._seed)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split (self._X_train_and_val, self._y_train_and_val, test_size=TEST_SIZE, random_state=self._seed)

        if NA_FILL_VAL is None:
            fill_nan = Imputer (missing_values=np.nan, strategy='mean', axis=1)
            self._X_train = fill_nan.fit_transform (self._X_train)
            self._X_val = fill_nan.transform (self._X_val)
            self._X_test = fill_nan.transform (self._X_test)
        else:
            self._X_train = self._X_train.fillna (value=NA_FILL_VAL)
            self._X_val = self._X_val.fillna (value=NA_FILL_VAL)
            self._X_test = self._X_test.fillna (value=NA_FILL_VAL)

        if scale:
            x_scaler = StandardScaler ()
            self._X_train_scaled = x_scaler.fit_transform (self._X_train)
            self._X_val_scaled = x_scaler.fit_transform (self._X_val)
            self._X_test_scaled = x_scaler.fit_transform (self._X_test)

            self._y_scaler = StandardScaler ()
            self._y_train_scaled = self._y_scaler.fit_transform (self._y_train)
        else:
            self._X_train_scaled = self._X_train
            self._X_val_scaled = self._X_val
            self._X_test_scaled = self._X_test
            
            self.y_train_scaled = self.y_train
        
    
    def print_summary (self, y_preds, y_acts, info):
        print ("Predictions vs actual values:" + info)
        print (["{0:.1f}".format(y_pred) for y_pred in y_preds])
        #print ("-----------------------------------------------------------")
        print (["{0:.1f}".format(y_act) for y_act in y_acts])
        #print ("-----------------------------------------------------------")
        print ("MSE: " + str (((y_preds - y_acts) ** 2).mean (axis=None)))
        print ("MAD: " + str ((abs(y_preds - y_acts)).mean (axis=None)))


    def fit_and_summarize_ann_model (self):
        
        self._model = MLPRegressor (solver='lbfgs', alpha=100)
        y_preds = None
        self._model.fit (self._X_train_scaled, np.ravel (self._y_train_scaled))
        y_preds_train = self._y_scaler.inverse_transform (self._model.predict (self._X_train_scaled))
        self.print_summary (y_preds_train, np.ravel (self._y_train), " - training")
        y_preds_val = self._y_scaler.inverse_transform (self._model.predict (self._X_val_scaled))
        self.print_summary (y_preds_val, np.ravel (self._y_val), " - validation")
        y_preds_test = self._y_scaler.inverse_transform (self._model.predict (self._X_test_scaled))
        self.print_summary (y_preds_test, np.ravel (self._y_test), " - test")


    def fit_and_summarize_rf_model (self):
        max_leaf_nodes_range = np.arange (2, 24, 7)
        max_features_range = np.arange (0.1, 1, 0.4)
        max_depth_range = np.arange (1, 22, 10)
        min_samples_split_range = np.arange (2, 7, 2)
        min_samples_leaf_range = np.arange (1, 11, 3)
        min_impurity_decrease_range = np.arange (0, 1, 0.3)
        #max_leaf_nodes_range = np.arange (1, 11, 1)
        param_grid = dict (max_leaf_nodes=max_leaf_nodes_range, max_features=max_features_range, max_depth=max_depth_range, \
                        min_samples_split=min_samples_split_range, min_samples_leaf=min_samples_leaf_range, \
                        min_impurity_decrease=min_impurity_decrease_range)
        #self._model = GridSearchCV (RandomForestRegressor (n_estimators=500, random_state=RNDM_SEED_2, verbose=0, warm_start=False), \
        #                param_grid=param_grid)
        self._model = RandomForestRegressor (n_estimators=500, random_state=RNDM_SEED_2, verbose=1,warm_start=False, max_leaf_nodes=5)
        y_preds = None
        self._model.fit (self._X_train_scaled, np.ravel (self._y_train_scaled))
        y_preds_train = self._y_scaler.inverse_transform (self._model.predict (self._X_train_scaled))
        self.print_summary (y_preds_train, np.ravel (self._y_train), " - training")
        y_preds_val = self._y_scaler.inverse_transform (self._model.predict (self._X_val_scaled))
        self.print_summary (y_preds_val, np.ravel (self._y_val), " - validation")
        y_preds_test = self._y_scaler.inverse_transform (self._model.predict (self._X_test_scaled))
        self.print_summary (y_preds_test, np.ravel (self._y_test), " - test")


def main ():
    data = load_data (refresh_live_sources=False)
    input_series, label_series = add_standard_columns (data)
    label_series = label_series.reset_index ()
    features_df = get_featurized_inputs (input_series, label_series, mths_to_combine=MTHS_TO_COMBINE)

    mdl = Model (days_prior=0, seed=RNDM_SEED_1)
    mdl.prepare_training_data (features_df, label_series, scale=True)
    #mdl.fit_and_summarize_ann_model ()
    mdl.fit_and_summarize_rf_model ()


if __name__ == "__main__":
    main()


