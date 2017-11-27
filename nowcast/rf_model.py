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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import util
import quandl_data

from util import *
from quandl_data import *

y_hat_frbny = None

#NA_FILL = 0
NA_FILL = -1e8
RNDM_SEED_1 = 7


def load_data (refresh=False):
    global y_hat_frbny
    if refresh:
        get_quandl_data ()
    else:
        for dc in QUANDL_DATA:
            file_name = Constant.DATA_DIR + "/" + dc + ".csv"
            if not os.path.isfile (file_name):
                #QUANDL_DATA [dc] ["data"] = pd.read_csv (file_name)
                get_quandl_data (set ([dc]))
            QUANDL_DATA [dc] ["data"] = pd.read_csv (file_name)
          
    # sn: needs manual updates - nowcasts not avaialable in feed format
    y_hat_frbny = pd.read_csv (Constant.DATA_DIR + "/ny_fed.csv")


def get_last_day_of_month (dt):
    return datetime.datetime (dt.year, dt.month, calendar.monthrange (dt.year, dt.month) [1])


# sn: this function returns X, y - the full data set since the beginning of y
# the function takes as argument the number of days prior to quarter end that we are
# predicting the GDP growth for
# x_df and y_df must have the column 'Date' and x_df must also have the column feat_col
def rf_featurize_series (days_to_qtr_end, x_df, y_df, num_days_per_period=91, num_years=2, fill_na_with=0, feat_col='Value'):
    X = []
    y = []
    quarters = []
    num_mths_to_combine = None
    if num_days_per_period >= 28 and num_days_per_period <= 31:
        num_mths_to_combine = 1
    if num_days_per_period >= 89 and num_days_per_period <= 92:
        num_mths_to_combine = 3
    if num_days_per_period >= 181 and num_days_per_period <= 184:
        num_mths_to_combine = 6
    if num_days_per_period >= 365 and num_days_per_period <= 366:
        num_mths_to_combine = 12

    DAYS_PER_YEAR = 365

    num_periods_per_series = None
    if not num_mths_to_combine:
        num_periods_per_series = num_years * int (DAYS_PER_YEAR/num_days_per_period)
    else:
        num_periods_per_series = (num_years * int (12 / num_mths_to_combine))

    for idx, row in y_df.iterrows ():
        qtr_end = row ['Date']
        #if qtr_end == datetime.datetime (2017,6,30):
        #    import pdb; pdb.set_trace ()
        x = [fill_na_with] * num_periods_per_series

        for period_idx in range (0, len (x)):
            tmp_df = x_df [(x_df ['Date'] <= qtr_end + relativedelta (days=-days_to_qtr_end)) & \
                                   (x_df ['Date'] <= get_last_day_of_month (qtr_end + relativedelta (months=-period_idx*num_mths_to_combine))) & \
                                   (x_df ['Date'] >  get_last_day_of_month (qtr_end + relativedelta (years=-num_years))) & \
                                   (x_df ['Date'] >  get_last_day_of_month (qtr_end + relativedelta (months=-(period_idx+1)*num_mths_to_combine)))]
            period_mean = tmp_df [feat_col].mean ()
            if not np.isnan (period_mean):
                x [period_idx] = tmp_df [feat_col].mean ()
        
        base = sum (x [1:])
        numer = sum (x [:1])
        if base != 0 and base != fill_na_with and numer != fill_na_with:
            x.append (numer / base)      
        else:
            x.append (fill_na_with)

        X.append (x)
        quarters.append (qtr_end)
        y.append (row ['Gross domestic product'])

    # import pdb; pdb.set_trace ()
    # sn: formatted print - keep this
    # print ('\n'.join ([', '.join ("{0:.1f}".format(d) for d in row) for row in X]))
    return np.array (X), np.array (y)


class Model (object):
    # sn: default is quarterly bucketing of data series to reduce noise that more frequent data usually has
    def __init__ (self, num_days_per_period=91, days_in_advance=0, use_scaling=False, seed=2):
        self._num_days_per_period = num_days_per_period
        self._days_in_advance = days_in_advance
        self._use_scaling = use_scaling
        self._seed = seed

        self._X = None
        self._y = None
        self._series_info = []
        self._model = None
        self._scaler = None
       

    # sn: input_series is a dict of lists with the series name as the key
    # while label_series is the output series (same for all input series)
    def prepare_series (self, input_series, label_series):
        fs = None
        y = None

        label_fname = 'featurized_series/gdp_growth.txt'
        if os.path.isfile (label_fname):
            self._y = np.loadtxt (label_fname)

        for series_name, series_data in input_series.items ():
            fname = 'featurized_series/' + series_name + '.txt'
            if os.path.isfile (fname):
                #fs = np.array (pd.read_csv (fname) ['Total'])
                fs = np.loadtxt (fname)
            else:
                if series_name == 'QTRLY_GDP_PCT_CHG_Bea':
                    fs, y = rf_featurize_series (1, series_data, label_series, \
                                self._num_days_per_period, fill_na_with=NA_FILL)
                else:
                    fs, y = rf_featurize_series (self._days_in_advance, series_data, label_series, \
                                self._num_days_per_period, fill_na_with=NA_FILL)
                np.savetxt (fname, fs)
         
            self._series_info.append ({'name': series_name})
            if self._use_scaling:
                self._series_info [-1] ['min_X'], self._series_info [-1] ['ptp_X'], fs \
                    = normalize_series (fs)
                self._series_info [-1] ['min_y'], self._series_info [-1] ['ptp_y'], y \
                    = normalize_series (y)

            if self._X is not None:
                self._X = np.concatenate ((self._X, fs), axis=1)
            else:
                self._X = fs

            if self._y is None and y is not None:
                self._y = y
                if not os.path.isfile (label_fname):
                    np.savetxt (label_fname, y)

        if self._X is not None:
            self._X = np.concatenate ((self._X, np.ones ((self._X.shape [0], 1))), axis=1)

        # sn: keeping validation and test data sizes at 10% each due to small data size
        X_train_and_val, self._X_test, y_train_and_val, self._y_test = \
            train_test_split (self._X, self._y, test_size=0.1, random_state=self._seed)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split (X_train_and_val, y_train_and_val, test_size=0.1, random_state=self._seed)


    def print_summary (self, y_preds, y_acts):

        print ("\nPredictions vs actual values:\n")
        print (["{0:.1f}".format(y_pred) for y_pred in y_preds])
        print ("-----------------------------------------------------------")
        print (["{0:.1f}".format(y_act) for y_act in y_acts])
        print ("-----------------------------------------------------------")
        print ("MSE: " + str (((y_preds - y_acts) ** 2).mean (axis=None)))
        print ("MAD: " + str ((abs(y_preds - y_acts)).mean (axis=None)))


    def fit_and_summarize_ridge_model (self, lambda_val, use_scaling=None):
        if not use_scaling:
            use_scaling = self._use_scaling
        self._model = Ridge (alpha=lambda_val, random_state=self._seed, fit_intercept=True, normalize=True)
        y_preds = None
        if use_scaling:
            self._scaler = sklearn.preprocessing.StandardScaler ()
            scaled_X_train = self._scaler.fit_transform (self._X_train)
            self._model.fit (scaled_X_train, self._y_train)
            y_preds = self._model.predict (self._scaler.transform (self._X_val))
        else:
            self._model.fit (self._X_train, self._y_train)
            y_preds = self._model.predict (self._X_val)

        self.print_summary (y_preds, self._y_val)


    def fit_and_summarize_random_forest_model (self):
        self._model = RandomForestRegressor ()
        y_preds = None
        self._model.fit (self._X_train, self._y_train)

        y_preds = self._model.predict (self._X_train)
        self.print_summary (y_preds, self._y_train)

        y_preds = self._model.predict (self._X_val)
        self.print_summary (y_preds, self._y_val)


    def print_test_data_performance (self):
        y_preds = None
        if self._scaler:
            y_preds = self._model.predict (self._scaler.transform (self._X_test))
        else:
            y_preds = self._model.predict (self._X_test)
        self.print_summary (y_preds, self._y_test)
 
   
def main ():
    load_data (refresh=False)
    # set up handles
    frbny_nowcast_df = y_hat_frbny
    input_series = {}
    label_series = None

    # sn: need a date that gives sufficient data points but also does not contain a lot of 'filler'
    # or missing data for the relevant, newer series
    cut_off_date = datetime.datetime.strptime ('1991-12-01', Constant.DATE_STR_FMT_1)

    for ds_name, ds_info in QUANDL_DATA.items ():
        df = ds_info ['data']
        if ds_name == "S&P_COMP_Yale":
            df ['Date'] = df ['Year'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
            input_series ['snp_earnings'] = df [['Date']].copy ()
        elif ds_name == "LONG_TERM_UNEMPL_FedReserve":
            df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
            df ['Date'] = df ['Date'].apply (lambda x: x + relativedelta (years=-10))
        else:
            df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))

        if ds_name == "PCE_Bea":
            df ['Value'] = df ['Personal consumption expenditures (PCE)']
        elif ds_name == "REAL_BLDG_COST_Yale":
            df ['Value'] = df ['Cost Index']
        elif ds_name == "S&P_COMP_Yale":
            df ['Value'] = df ['S&P Composite']
            input_series ['snp_earnings'] ['Value'] = df ['Real Earnings']
        elif ds_name == "INDIV_STK_MKT_CONF_Yale" or ds_name == "INSTI_STK_MKT_CONF_Yale":
            df ['Value'] = df ['Index Value']
        elif ds_name == "FIXED_CAP_CONS_Bea":
            df ['Value'] = df ['Consumption of fixed capital']
        elif ds_name == "QTRLY_GDP_PCT_CHG_Bea":
            df ['Value'] = df ['Gross domestic product']
            label_series = df [df ['Date'] >= cut_off_date]
        elif ds_name == "NMI_ISM":
            df ['Value'] = df ['Index']

        input_series [ds_name] = df

    rf = Model (num_days_per_period=365, days_in_advance=0, use_scaling=False, seed=RNDM_SEED_1)
    rf.prepare_series (input_series, label_series)
    rf.fit_and_summarize_random_forest_model ()
    rf.print_test_data_performance ()


if __name__ == "__main__":
    main()
