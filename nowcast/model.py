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
from dateutil.relativedelta import relativedelta
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
def featurize_series (days_to_qtr_end, x_df, y_df, num_days_per_period=91, num_years=2):
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
    ##data_pt_count = {}

    for idx, row in y_df.iterrows ():
        qtr_end = row ['Date']
        x = None
        if not num_mths_to_combine:
            x = [0] * num_years * int (DAYS_PER_YEAR/num_days_per_period)
        else:
            x = [0] * (num_years * int (12 / num_mths_to_combine))
        data_pt_count = x [:]
        for idx2, row2 in x_df.iterrows ():
            series_row_date = row2 ['Date']
            if (series_row_date <= qtr_end + timedelta (days=-days_to_qtr_end)) and \
                    (series_row_date > qtr_end + relativedelta (years=-num_years)):
                feature_col_idx = int (relativedelta(qtr_end, series_row_date).months / num_mths_to_combine) + \
                                    int (relativedelta (qtr_end, series_row_date).years * (12 / num_mths_to_combine))
                #import pdb;pdb.set_trace ()
                #if feature_col_idx >= len (x):
                #    feature_col_idx = len (x) - 1
                x [feature_col_idx] += row2 ['Total']
                data_pt_count [feature_col_idx] += 1

        for idx in range (len (x)):
            if data_pt_count [idx] == 0:
                if idx + 1 < len (x) and data_pt_count [idx + 1] > 0:
                    x [idx] = x [idx + 1] / data_pt_count [idx + 1]
                else:
                    import pdb; pdb.set_trace ()
                    raise ('Data error to be fixed! No data found for qtr ending ' + str (qtr_end) + ' in col ' + str (idx))
            else:
                # sn: use quarterly averages
                x [idx] = x [idx] / data_pt_count [idx]
       
        X.append (x)
        quarters.append (qtr_end)
        y.append (row ['Gross domestic product'])

    # import pdb; pdb.set_trace ()
    # sn: formatted print - keep this
    # print ('\n'.join ([', '.join ("{0:.1f}".format(d) for d in row) for row in X]))
    return np.array (X), np.array (y)

class LinearRegressionModel (object):
    # sn: default is quarterly bucketing of data series to reduce noise that more frequent data usually has
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

        label_fname = 'featurized_series/gdp_growth.txt'
        if os.path.isfile (label_fname):
            self._y = np.loadtxt (label_fname)

        for series_name, series_data in input_series.items ():
            fname = 'featurized_series/' + series_name + '.txt'
            if os.path.isfile (fname):
                #fs = np.array (pd.read_csv (fname) ['Total'])
                fs = np.loadtxt (fname)
            else:
                if series_name == 'gdp':
                    fs, y = featurize_series (1, series_data, label_series, self._num_days_per_period)
                else:
                    fs, y = featurize_series (self._days_in_advance, series_data, label_series, self._num_days_per_period)
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
    
    pce_df ['Total'] = pce_df ['Personal consumption expenditures (PCE)'] 
    wti_df ['Total'] = wti_df ['Value']
    unempl_df ['Total'] = unempl_df ['Value']
    avg30ymtg_df ['Total'] = avg30ymtg_df ['Value']
    snp_df ['Total'] = snp_df ['S&P Composite']
    indiv_conf_df ['Total'] = indiv_conf_df ['Index Value']
    insti_conf_df ['Total'] = insti_conf_df ['Index Value']

    act_gdp_df ['Date'] = act_gdp_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    act_gdp_df ['Total'] = act_gdp_df ['Gross domestic product']

    # sn: this will be n years after the availability of the latest beginning of any used series
    # currently 2 years after availability of first stock market confidence measures
    cut_off_date = datetime.datetime.strptime ('1991-12-01', Constant.DATE_STR_FMT_1)

    input_series = {
        'pce': pce_df 
        ,'wti': wti_df
        ,'unempl': unempl_df
        ,'avg30ymtg': avg30ymtg_df
        ,'snp': snp_df
        ,'gdp': act_gdp_df
        #,'indiv_conf': indiv_conf_df
        #,'insti_conf': insti_conf_df
    }
    label_series = act_gdp_df [act_gdp_df ['Date'] >= cut_off_date]

    lrm = LinearRegressionModel (num_days_per_period=91, days_in_advance=0, use_scaling=False, seed=2)
    lrm.prepare_series (input_series, label_series)

    lrm.fit_and_summarize_ridge_model (1)
    lrm.print_test_data_performance ()

    # featurize and normalize the data
    X = None
    y = None


if __name__ == "__main__":
    main()
