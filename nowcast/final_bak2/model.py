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
RNDM_SEED_2 = 3
TEST_SIZE = 0.10
NA_FILL_VAL=None

MTHS_TO_COMBINE=3
TS_DATES = {
    'VAL_START': datetime.datetime (2014,1,1),
    'VAL_END': datetime.datetime (2016,1,1),
    'TEST_START': datetime.datetime (2016,1,1),
    'TEST_END': datetime.datetime (2018,1,1)
}

class Model (object):
    # sn: default is quarterly bucketing of data series to reduce noise that more frequent data usually has
    def __init__ (self, days_prior=0, seed=RNDM_SEED_1):
        self._days_prior = days_prior
        self._seed = seed

        self._X = None
        self._y = None
        self._model = None
        self._y_scaler = None
        self._X_imputer = None
        self._X_scaler = None

    def prepare_training_data (self, x_df, y_df, scale=True):
        X = x_df.drop (columns=['Date'], axis=1).as_matrix ()
        y = y_df.drop (columns=['Date'], axis=1).as_matrix ()

        self._X_train_and_val, self._X_test, self._y_train_and_val, self._y_test = \
            train_test_split (X, y, test_size=TEST_SIZE, random_state=self._seed)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split (self._X_train_and_val, self._y_train_and_val, test_size=TEST_SIZE, random_state=self._seed)

        self._X_train = np.delete (self._X_train, 0, axis=1)
        self._X_train_and_val = np.delete (self._X_train_and_val, 0, axis=1)
        self._X_val = np.delete (self._X_val, 0, axis=1)
        self._X_test = np.delete (self._X_test, 0, axis=1)

        self._y_train = np.delete (self._y_train, 0, axis=1)
        self._y_train_and_val = np.delete (self._y_train_and_val, 0, axis=1)
        self._y_val = np.delete (self._y_val, 0, axis=1)
        self._y_test = np.delete (self._y_test, 0, axis=1)

        if NA_FILL_VAL is None:
            fill_nan = Imputer (missing_values=np.nan, strategy='mean', axis=0)
            self._X_train = fill_nan.fit_transform (self._X_train)
            self._X_imputer = fill_nan
            self._X_val = fill_nan.transform (self._X_val)
            self._X_test = fill_nan.transform (self._X_test)
        else:
            self._X_train = self._X_train.fillna (value=NA_FILL_VAL)
            self._X_val = self._X_val.fillna (value=NA_FILL_VAL)
            self._X_test = self._X_test.fillna (value=NA_FILL_VAL)

        if scale:
            x_scaler = StandardScaler ()
            self._X_train_scaled = x_scaler.fit_transform (self._X_train)
            self._X_scaler = x_scaler
            self._X_val_scaled = x_scaler.transform (self._X_val)
            self._X_test_scaled = x_scaler.transform (self._X_test)

            self._y_scaler = StandardScaler ()
            self._y_train_scaled = self._y_scaler.fit_transform (self._y_train)
        else:
            self._X_train_scaled = self._X_train
            self._X_val_scaled = self._X_val
            self._X_test_scaled = self._X_test
            
            self.y_train_scaled = self.y_train
        
    
    def prepare_training_data_ts (self, x_df, y_df, scale=True):
        X =  x_df
        y = y_df
        ''''
        X = x_df.drop (columns=['Date'], axis=1).as_matrix ()
        y = y_df.drop (columns=['Date'], axis=1).as_matrix ()

        self._X_train_and_val, self._X_test, self._y_train_and_val, self._y_test = \
            train_test_split (X, y, test_size=TEST_SIZE, random_state=self._seed)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split (self._X_train_and_val, self._y_train_and_val, test_size=TEST_SIZE, random_state=self._seed)
        '''
        #import pdb; pdb.set_trace ()
        self._X_train = X [((X ['Date'] < TS_DATES ['VAL_START']) | (X ['Date'] >= TS_DATES ['TEST_END']))]
        self._y_train = y [((y ['Date'] < TS_DATES ['VAL_START']) | (y ['Date'] >= TS_DATES ['TEST_END']))]
        self._X_train_and_val = X [((X ['Date'] >= TS_DATES ['VAL_START']) & (X ['Date'] < TS_DATES ['VAL_END']) | \
                                   (X ['Date'] >= TS_DATES ['TEST_START']) & (X ['Date'] < TS_DATES ['TEST_END']))]
        self._y_train_and_val = y [((y ['Date'] >= TS_DATES ['VAL_START']) & (y ['Date'] < TS_DATES ['VAL_END']) & \
                                   (y ['Date'] >= TS_DATES ['TEST_START']) & (y ['Date'] < TS_DATES ['TEST_END']))]
        self._X_val = X [((X ['Date'] >= TS_DATES ['VAL_START']) & (X ['Date'] < TS_DATES ['VAL_END']))]
        self._y_val = y [((y ['Date'] >= TS_DATES ['VAL_START']) & (y ['Date'] < TS_DATES ['VAL_END']))]
        self._X_test = X [((X ['Date'] >= TS_DATES ['TEST_START']) & (X ['Date'] < TS_DATES ['TEST_END']))]
        self._y_test = y [((y ['Date'] >= TS_DATES ['TEST_START']) & (y ['Date'] < TS_DATES ['TEST_END']))]
        
        self._X_train = self._X_train.drop (columns=['Date'], axis=1).as_matrix ()
        self._X_train_and_val = self._X_train_and_val.drop (columns=['Date'], axis=1).as_matrix ()
        self._X_val = self._X_val.drop (columns=['Date'], axis=1).as_matrix ()
        self._X_test = self._X_test.drop (columns=['Date'], axis=1).as_matrix ()

        self._X_train = np.delete (self._X_train, 0, axis=1)
        self._X_train_and_val = np.delete (self._X_train_and_val, 0, axis=1)
        self._X_val = np.delete (self._X_val, 0, axis=1)
        self._X_test = np.delete (self._X_test, 0, axis=1)

        self._y_train = self._y_train.drop (columns=['Date'], axis=1).as_matrix ()
        self._y_train_and_val = self._y_train_and_val.drop (columns=['Date'], axis=1).as_matrix ()
        self._y_val = self._y_val.drop (columns=['Date'], axis=1).as_matrix ()
        self._y_test = self._y_test.drop (columns=['Date'], axis=1).as_matrix ()

        self._y_train = np.delete (self._y_train, 0, axis=1)
        self._y_train_and_val = np.delete (self._y_train_and_val, 0, axis=1)
        self._y_val = np.delete (self._y_val, 0, axis=1)
        self._y_test = np.delete (self._y_test, 0, axis=1)

        if NA_FILL_VAL is None:
            fill_nan = Imputer (missing_values=np.nan, strategy='mean', axis=0)
            self._X_train = fill_nan.fit_transform (self._X_train)
            self._X_imputer = fill_nan
            self._X_val = fill_nan.transform (self._X_val)
            self._X_test = fill_nan.transform (self._X_test)
        else:
            self._X_train = self._X_train.fillna (value=NA_FILL_VAL)
            self._X_val = self._X_val.fillna (value=NA_FILL_VAL)
            self._X_test = self._X_test.fillna (value=NA_FILL_VAL)

        if scale:
            x_scaler = StandardScaler ()
            self._X_train_scaled = x_scaler.fit_transform (self._X_train)
            self._X_scaler = x_scaler
            self._X_val_scaled = x_scaler.transform (self._X_val)
            self._X_test_scaled = x_scaler.transform (self._X_test)

            self._y_scaler = StandardScaler ()
            self._y_train_scaled = self._y_scaler.fit_transform (self._y_train)
        else:
            self._X_train_scaled = self._X_train
            self._X_val_scaled = self._X_val
            self._X_test_scaled = self._X_test
            
            self.y_train_scaled = self.y_train
        
    
    def prepare_comparison_data (self, x_df, scale=True):
        X = np.delete (x_df.drop (columns=['Date', 'y_ny_fed_prediction', 'Gross domestic product'], axis=1).as_matrix (), 0, axis=1)
        X = np.delete (X, -1, axis=1)
        eval_df = X
        if NA_FILL_VAL is None:
            eval_df = self._X_imputer.transform (eval_df)
        else:
            eval_df = eval_df.fillna (value=NA_FILL_VAL)
            
        if scale:
            eval_df = self._X_scaler.transform (eval_df)

        return eval_df
    
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
        '''
        max_leaf_nodes_range = np.arange (2, 24, 7)
        max_features_range = np.arange (0.1, 1, 0.4)
        max_depth_range = np.arange (1, 22, 10)
        min_samples_split_range = np.arange (2, 7, 2)
        min_samples_leaf_range = np.arange (1, 11, 3)
        min_impurity_decrease_range = np.arange (0, 1, 0.3)
        param_grid = dict (max_leaf_nodes=max_leaf_nodes_range, max_features=max_features_range, max_depth=max_depth_range, \
                        min_samples_split=min_samples_split_range, min_samples_leaf=min_samples_leaf_range, \
                        min_impurity_decrease=min_impurity_decrease_range)
        self._model = GridSearchCV (RandomForestRegressor (n_estimators=500, random_state=RNDM_SEED_2, verbose=0, warm_start=False), \
                        param_grid=param_grid)
        '''
       
        '''
        self._model = RandomForestRegressor (n_estimators=500, random_state=RNDM_SEED_2, verbose=1,warm_start=False, max_leaf_nodes=7, \
                        max_features=1.0, max_depth=2, min_samples_split=7, min_samples_leaf=4, min_impurity_decrease=0)
        '''
        
        self._model = RandomForestRegressor (n_estimators=500, random_state=RNDM_SEED_2, verbose=1)
        
        y_preds = None
        self._model.fit (self._X_train_scaled, np.ravel (self._y_train_scaled))
        y_preds_train = self._y_scaler.inverse_transform (self._model.predict (self._X_train_scaled))
        self.print_summary (y_preds_train, np.ravel (self._y_train), " - training")
        y_preds_val = self._y_scaler.inverse_transform (self._model.predict (self._X_val_scaled))
        self.print_summary (y_preds_val, np.ravel (self._y_val), " - validation")
        y_preds_test = self._y_scaler.inverse_transform (self._model.predict (self._X_test_scaled))
        self.print_summary (y_preds_test, np.ravel (self._y_test), " - test")


    def compare_with_benchmark (self, input_series, label_series):
        benchmark_df = pd.read_csv ('nowcast_benchmarks/ny_fed.csv')
        benchmark_df ['Value'] = benchmark_df ['Value'].replace (to_replace='-', value=np.nan) 
        benchmark_df ['Value'] = np.asfarray (benchmark_df ['Value'])
        x_values = None
        y_ny_fed_predictions = []
        count = 0
        # sn: load input vectors
        if not os.path.isfile ('nowcast_benchmarks/x_values.csv'):
            for index, row in benchmark_df.iterrows ():
                if np.isfinite (row ['Value']):
                    prediction_date = datetime.datetime.strptime (row ['Date'], Constant.DATE_STR_FMT_2)
                    qtr_end = datetime.datetime.strptime (row ['TargetPeriodDate'], Constant.DATE_STR_FMT_1)
                    # sn: revisit to see if the logic is good for data released past the quarter end but before the official gdp release
                    if prediction_date < qtr_end:
                        days_prior = (qtr_end - prediction_date).days
                        features_df = get_featurized_inputs (input_series, label_series [label_series ['Date']==qtr_end], days_prior=days_prior)
                        features_df ['y_ny_fed_prediction'] = row ['Value']
                        label_df = label_series [label_series ['Date']==qtr_end] ['Gross domestic product'].copy ().reset_index ()
                        features_df = pd.concat ([features_df, label_df], axis=1)
                        if x_values is None:
                            x_values = features_df
                        else:
                            x_values = pd.concat ([x_values, features_df], axis=0)
                        count += 1
            x_values.to_csv ('nowcast_benchmarks/x_values.csv', index=False)
        else:
            x_values = pd.read_csv ('nowcast_benchmarks/x_values.csv')

        y_ny_fed_predictions = x_values ['y_ny_fed_prediction']
        y_actuals = x_values ['Gross domestic product']
        x_eval = self.prepare_comparison_data (x_values, scale=True)
        y_preds = self._y_scaler.inverse_transform (self._model.predict (x_eval))
        self.print_summary (y_preds, np.ravel (y_actuals), " - NY Fed data points - our model against actual GDP growth")
        self.print_summary (y_ny_fed_predictions, np.ravel (y_actuals), " - NY Fed data points - Fed nowcast against actual GDP growth")

def main ():
    data = load_data (refresh_live_sources=False)
    input_series, label_series = add_standard_columns (data)
    label_series = label_series.reset_index ()
    features_df = get_featurized_inputs (input_series, label_series, mths_to_combine=MTHS_TO_COMBINE)

    mdl = Model (days_prior=0, seed=RNDM_SEED_1)
    mdl.prepare_training_data_ts (features_df, label_series, scale=True)
    #mdl.fit_and_summarize_ann_model ()
    mdl.fit_and_summarize_rf_model ()

    mdl.compare_with_benchmark (input_series, label_series)

if __name__ == "__main__":
    main()


