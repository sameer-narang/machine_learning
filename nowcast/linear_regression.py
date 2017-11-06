#!/usr/bin/python3.5

import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
import sklearn

from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import util
import quandl_data

from util import *
from quandl_data import *

y_hat_frbny = None

def load_data (refresh=False):
    global y_hat_frbny
    if refresh:
        get_quandl_data ()
    else:
        for dc in QUANDL_DATA:
            file_name = Constant.DATA_DIR + "/" + dc + ".csv"
            if os.path.isfile (file_name):
                QUANDL_DATA [dc] ["data"] = pd.read_csv (file_name)
          
    # sn: needs manual updates - nowcasts not avaialable in feed format
    y_hat_frbny = pd.read_csv (Constant.DATA_DIR + "/ny_fed.csv")


# sn: this function returns X, y - the full data set since the beginning of y
# each day in the prior 2 years is a feature
# the function takes as argument the number of days prior to quarter end that we are
# predicting the GDP growth for
# x_df and y_df must have the column 'Date' and x_df must also have the column 'Total'
def featurize_series (days_to_qtr_end, x_df, y_df, num_days_per_period=1, num_years=2):
    X = []
    y = []
    quarters = []
    DAYS_PER_YEAR = 365

    for idx, row in y_df.iterrows ():
        #quarters.append (convert_yyyymmdd_str_to_qtr(row ['Date']))
        #qtr_end = datetime.datetime.strptime (row ['Date'], fmt)
        qtr_end = row ['Date']
        x = [0] * num_years * int (DAYS_PER_YEAR/num_days_per_period)
        for idx2, row2 in x_df.iterrows ():
            series_row_date = row2 ['Date']
            if (series_row_date <= qtr_end + timedelta (days=-days_to_qtr_end)) and \
                    (series_row_date > qtr_end + timedelta (days=-2 * num_years * DAYS_PER_YEAR)):
                feature_col_idx = int ((qtr_end - series_row_date).days / num_days_per_period)
                if feature_col_idx >= len (x):
                    feature_col_idx = len (x) - 1
                x [feature_col_idx] = row2 ['Total']
       
        X.append (x)
        quarters.append (qtr_end)
        y.append (row ['Gross domestic product'])

    return np.array (X), np.array (y)

class LinearRegressionModel (object):
    # sn: default is quarterly bucketing of data series to reduce noise
    def __init__ (self, num_days_per_period=91, days_in_advance=0, use_scaling=False, seed=2):
        self._num_days_per_period = num_days_per_period
        self._days_in_advance = days_in_advance
        self._use_scaling = use_scaling
        self._seed = seed

        self._X = None
        self._y = None
        self._series_info = []
        self._ridgeRegModel = None
       
    # sn: input_series is a dict of lists with the series name as the key
    # while label_series is the output series (same for all input series)
    def prepare_series (self, input_series, label_series):
        fs = None
        y = None
        for series_name, series_data in input_series.items ():
            fs, y = featurize_series (self._days_in_advance, series_data, label_series, self._num_days_per_period)
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

            if self._y is None:
                self._y = y

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


    def fit_and_summarize_ridge_model (self, lambda_val):
        self._ridgeRegModel = Ridge (alpha=lambda_val, random_state=self._seed)
        self._ridgeRegModel.fit (self._X_train, self._y_train)

        y_preds = self._ridgeRegModel.predict (self._X_val)
        self.print_summary (y_preds, self._y_val)

    def print_test_data_performance (self):
        y_preds = self._ridgeRegModel.predict (self._X_test)
        self.print_summary (y_preds, self._y_test)
    


def main ():
    load_data (refresh=False)
    # set up handles
    frbny_nowcast_df = y_hat_frbny

    pce_df = QUANDL_DATA ["PCE_Bea"] ["data"]
    wti_df = QUANDL_DATA ["WTI_DeptOfEnergy"] ["data"]
    unempl_df = QUANDL_DATA ["UNEMPLOYMENT_FedReserve"] ["data"]
    avg30ymtg_df = QUANDL_DATA ["AVG_30Y_MTGG_FreddieMac"] ["data"]
    #bldg_cost_df = QUANDL_DATA ["REAL_BLDG_COST_Yale"] ["data"]
    snp_df = QUANDL_DATA ["S&P_COMP_Yale"] ["data"]
    indiv_conf_df = QUANDL_DATA ["INDIV_STK_MKT_CONF_Yale"] ["data"]
    insti_conf_df = QUANDL_DATA ["INSTI_STK_MKT_CONF_Yale"] ["data"]
    act_gdp_df = QUANDL_DATA ["QTRLY_GDP_PCT_CHG_Bea"] ["data"]

    pce_df ['Date'] = pce_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    wti_df ['Date'] = wti_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    unempl_df ['Date'] = unempl_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    avg30ymtg_df ['Date'] = avg30ymtg_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    snp_df ['Date'] = snp_df ['Year'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    indiv_conf_df ['Date'] = indiv_conf_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    insti_conf_df ['Date'] = insti_conf_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    
    pce_df ['Total'] = pce_df ['Personal consumption expenditures (PCE)'] + pce_df ['Goods'] + \
                       pce_df [':Durable goods'] + pce_df [':Nondurable goods'] + pce_df ['Services'] + \
                       pce_df ['PCE excluding food and energy (Addenda)'] + pce_df ['Food (Addenda)'] + \
                       pce_df ['Energy goods and services (Addenda)']
    wti_df ['Total'] = wti_df ['Value']
    unempl_df ['Total'] = unempl_df ['Value']
    avg30ymtg_df ['Total'] = avg30ymtg_df ['Value']
    snp_df ['Total'] = snp_df ['S&P Composite']
    indiv_conf_df ['Total'] = indiv_conf_df ['Index Value']
    insti_conf_df ['Total'] = insti_conf_df ['Index Value']
    act_gdp_df ['Date'] = act_gdp_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))

    # sn: this will be n years after the availability of the latest beginning of any used series
    # currently 2 years after availability of first stock market confidence measures
    cut_off_date = datetime.datetime.strptime ('1991-12-01', Constant.DATE_STR_FMT_1)

    input_series = {
        'pce': pce_df,
        'wti': wti_df,
        'unempl': unempl_df,
        'avg30ymtg': avg30ymtg_df,
        'snp': snp_df,
        'indiv_conf': indiv_conf_df,
        'insti_conf': insti_conf_df
    }
    label_series = act_gdp_df [act_gdp_df ['Date'] >= cut_off_date]

    lrm = LinearRegressionModel (num_days_per_period=91, days_in_advance=0, use_scaling=False, seed=2)
    lrm.prepare_series (input_series, label_series)

    lrm.fit_and_summarize_ridge_model (1)
    lrm.print_test_data_performance ()

    # featurize and normalize the data
    X = None
    y = None


    '''
    X1, y1 = featurize_series (0, pce_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X2, y2 = featurize_series (0, wti_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X3, y3 = featurize_series (0, unempl_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X4, y4 = featurize_series (0, avg30ymtg_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X5, y5 = featurize_series (0, snp_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X6, y6 = featurize_series (0, indiv_conf_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)
    X7, y7 = featurize_series (0, insti_conf_df, \
        act_gdp_df [act_gdp_df ['Date'] >= cut_off_date], num_days_per_period=91)

    scale_data = False
    
    if scale_data:
        min_y, ptp_y, scaled_y1 = normalize_series (y1)
        min_X1, ptp_X1, scaled_X1 = normalize_series (X1)
        min_X2, ptp_X2, scaled_X2 = normalize_series (X2)

        X = np.concatenate ((scaled_X1, scaled_X2), axis=1)
        y = scaled_y1
    else:
        X = np.concatenate ((X1, X2), axis=1)
        y = y1

    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=2)


    regr = LinearRegression ()
    regr.fit (X_train, y_train)
    y_preds = regr.predict (X_test)

    print ("Predictions vs actual values:\n")
    print (["{0:.1f}".format(y_pred) for y_pred in y_preds])
    print ("-----------------------------------------------------------")
    print (["{0:.1f}".format(y_act) for y_act in y_test])
    print ("-----------------------------------------------------------")
    print ("MSE: " + str (((y_preds - y_test) ** 2).mean (axis=None)))
    '''
    
if __name__ == "__main__":
    main()
