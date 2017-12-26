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

RNDM_SEED_1 = 1
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
        self._X_train_and_val, self._X_test, self._y_train_and_val, self._y_test = \
            train_test_split (x_df, y_df, test_size=TEST_SIZE, random_state=self._seed)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split (self._X_train_and_val, self._y_train_and_val, test_size=TEST_SIZE, random_state=self._seed)

        if NA_FILL_VAL is None:
            fill_nan = Imputer (missing_values=np.nan, strategy='mean', axis=1)
            self._X_train = pd.DataFrame (fill_nan.fit_transform (self._X_train))
            self._X_val = pd.DataFrame (fill_nan.transform (self._X_val))
            self._X_test = pd.DataFrame (fill_nan.transform (self._X_test))
        else:
            x_df = x_df.fillna (value=NA_FILL_VAL)
    
        if scale:
            self.X_train_scaled = pd.DataFrame ({})
            self.X_val_scaled = pd.DataFrame ({})
            self.X_test_scaled = pd.DataFrame ({})
        
            self._y_scaler = StandardScaler ()
            self.y_train_scaled = pd.DataFrame ({})
            self.y_val_scaled = pd.DataFrame ({})
            self.y_test_scaled = pd.DataFrame ({})
            self._y_train_scaled ['Date'] = self._y_train ['Date']
            self._y_train_scaled ['Gross domestic product'] = \
                self._y_scaler.fit_transform (self._y_train ['Gross domestic product'])
            '''
            self._y_val_scaled ['Date'] = self._y_val ['Date']
            self._y_val_scaled ['Gross domestic product'] = \
                self._y_scaler.transform (self._y_val ['Gross domestic product'])
            self._y_test_scaled ['Date'] = self._y_test ['Date']
            self._y_test_scaled ['Gross domestic product'] = \
                self._y_scaler.transform (self._y_test ['Gross domestic product'])
            '''
        else:
            self.X_train_scaled = self.X_train
            self.X_val_scaled = self.X_val
            self.X_test_scaled = self.X_test
            
            self.y_train_scaled = self.y_train
        
        for col in x_df:
            if col != 'Date':
                if scale:
                    x_scaler = StandardScaler ()
                    self._X_train_scaled [col] = x_scaler.fit_transform (self._X_train [col])
                    self._X_val_scaled [col] = x_scaler.transform (self._X_val [col])
                    self._X_test_scaled [col] = x_scaler.transform (self._X_test [col])
            else:
                self._X_train_scaled [col] = self._X_train [col]
                self._X_val_scaled [col] = self._X_val [col]
                self._X_test_scaled [col] = self._X_test [col]

    
    def print_summary (self, y_preds, y_acts, info):
        print ("\nPredictions vs actual values:" + info)
        print (["{0:.1f}".format(y_pred) for y_pred in y_preds])
        #print ("-----------------------------------------------------------")
        print (["{0:.1f}".format(y_act) for y_act in y_acts])
        #print ("-----------------------------------------------------------")
        print ("MSE: " + str (((y_preds - y_acts) ** 2).mean (axis=None)))
        print ("MAD: " + str ((abs(y_preds - y_acts)).mean (axis=None)))


    def fit_and_summarize_ann_model (self):
        self._X_train_scaled.drop ('Date')
        self._X_val_scaled.drop ('Date')
        self._X_test_scaled.drop ('Date')

        self._y_train_scaled.drop ('Date')

        self._model = MLPRegressor (solver='lbfgs', alpha=100)
        y_preds = None
        self._model.fit (self._X_train_scaled, self._y_train_scaled)
        y_preds_train = self._y_scaler.inverse_transform (_self._model.predict (self._X_train_scaled))
        self.print_summary (y_preds_train, self._y_train, " - training")
        y_preds_val = self._y_scaler.inverse_transform (self._model.predict (self._X_val_scaled))
        self.print_summary (y_preds_val, self._y_val, " - validation")
        y_preds_test = self._y_scaler.inverse_transform (self._model.predict (self._X_test_scaled))
        self.print_summary (y_preds_test, self._y_test, " - test")


def main ():
    data = load_data (refresh_live_sources=False)
    input_series, label_series = add_standard_columns (data)
    features_df = get_featurized_inputs (input_series, label_series, mths_to_combine=MTHS_TO_COMBINE)

    mdl = Model (days_prior=0, seed=RNDM_SEED_1)
    mdl.prepare_training_data (features_df, label_series, scale=True)
    mdl.fit_and_summarize_ann_model ()


if __name__ == "__main__":
    main()


