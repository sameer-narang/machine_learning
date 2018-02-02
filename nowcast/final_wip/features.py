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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import util
import raw_data

from util import *
from raw_data import *

y_hat_frbny = None

cut_off_date = datetime.datetime.strptime ('1991-12-01', Constant.DATE_STR_FMT_1)


def get_last_day_of_month (dt):
    return datetime.datetime (dt.year, dt.month, calendar.monthrange (dt.year, dt.month) [1])


def add_standard_columns (data):
    input_series = {}
    label_series = None

    for ds_name, ds_info in data.items ():
        df = ds_info ['data']
        if ds_name == "YALE_SPCOMP":
            df ['Date'] = df ['Year'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
            input_series ['snp_earnings'] = df [['Date']].copy ()
        elif ds_name == "FRED_NROU":
            df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
            df ['Date'] = df ['Date'].apply (lambda x: x + relativedelta (years=-10))
        elif ds_name == "CHRIS_CBOE_VX1":
            df ['Date'] = df ['Trade Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
        elif ds_name == "ADP_EMPL_NONFARM_PRI":
            df ['Date'] = df ['Month'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
        else:
            df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))

        if ds_name == "BEA_T20805_M":
            df ['Value'] = df ['Personal consumption expenditures (PCE)']
        elif ds_name == "YALE_RBCI":
            df ['Value'] = df ['Cost Index']
        elif ds_name == "YALE_SPCOMP":
            df ['Value'] = df ['S&P Composite']
            input_series ['snp_earnings'] ['Value'] = df ['Real Earnings']
        elif ds_name == "YALE_US_CONF_INDEX_VAL_INDIV" or ds_name == "YALE_US_CONF_INDEX_VAL_INST":
            df ['Value'] = df ['Index Value']
        elif ds_name == "BEA_T70500_Q":
            df ['Value'] = df ['Consumption of fixed capital']
        elif ds_name == "BEA_T10107_Q":
            df ['Value'] = df ['Gross domestic product']
            label_series = df [df ['Date'] >= cut_off_date]
            label_series = label_series [['Date', 'Gross domestic product']]
        elif ds_name == "ISM_NONMAN_NMI":
            df ['Value'] = df ['Index']
        elif ds_name == "CHRIS_CBOE_VX1":
            df ['Value'] = df ['Close']
        elif ds_name == "ADP_EMPL_NONFARM_PRI":
            df ['Value'] = df ['1-49'] + df ['50-499'] + df ['500+']

        input_series [ds_name] = df
 
    validate_input_data (data)
    return input_series, label_series


def validate_input_data (data):
    print ("Checking all input series for 'Value' and 'Date' columns")
    for ds_name, ds_info in data.items ():
        df = ds_info ['data']
        if 'Value' not in df.columns:
            raise Exception ("Data series " + ds_name + " does not have the standard required column 'Value'")
        if 'Date' not in df.columns:
            raise Exception ("Data series " + ds_name + " does not have the standard required column 'Date'")


def get_featurized_inputs (input_series, label_series, mths_to_combine=3, days_prior=0):
    features_df = pd.DataFrame (label_series ['Date'].copy ()).reset_index ()
    for series_name, series_data in input_series.items ():
        fname = 'featurized_series/' + series_name + '_' + str (mths_to_combine)  + '.csv'

        df = None
        if days_prior == 0 and os.path.isfile (fname):
            df = pd.read_csv (fname)
        else:
            if series_name == 'BEA_T10107_Q':
                df = featurize_df (series_data, label_series, mths_to_combine=mths_to_combine, days_prior=1)
            else:
                df = featurize_df (series_data, label_series, mths_to_combine=mths_to_combine,days_prior=days_prior)
            if days_prior == 0:
                df.to_csv (fname, index=False)

        col_name_map = {}
        df = df.drop (['Date'], axis=1)
        for col in df:
            if col != 'Date':
                col_name_map [col] = series_name + "__" + col
        df = df.rename (col_name_map, axis='columns')
        features_df = pd.concat ([features_df, df], axis=1)

    return features_df


def featurize_df (x_df, y_df, mths_to_combine, days_prior=0):
    X = []
    y = []
    quarters = []
    feat_col = 'Value'
    num_years = 2
    result = {'Date': [], 'yoy_diff': [], 'yoy_ratio': [], 'qoq_diff': [], 'qoq_ratio': []}
    #result = {'Date': [], 'qoq_diff': [], 'qoq_ratio': []}
    num_periods_per_series = num_years * int (12/ mths_to_combine)
    x = [np.nan] * num_periods_per_series
    for i in range (0, num_periods_per_series):
        result [str (i*mths_to_combine) + '_mths_prior'] = []

    if mths_to_combine < 12:
        result ['past_year_1'] = []
        result ['past_year_2'] = []
    
    if mths_to_combine < 12:
        result ['past_qtr_1'] = []
        result ['past_qtr_2'] = []
    
    result  ['q0'] = []
    result  ['q1'] = []
    result  ['q2'] = []
    result  ['q3'] = []
    result  ['q4'] = []

    for idx, row in y_df.iterrows ():
        qtr_end = row ['Date']
        result ['Date'].append (qtr_end)
        for period_idx in range (0, num_periods_per_series):
            tmp_df = \
                x_df [(x_df ['Date'] <= qtr_end + relativedelta (days=-days_prior)) & \
                (x_df ['Date'] <= get_last_day_of_month (qtr_end + relativedelta (months=-period_idx*mths_to_combine))) & \
                (x_df ['Date'] >  get_last_day_of_month (qtr_end + relativedelta (months=-(period_idx+1)*mths_to_combine)))]
            period_mean = tmp_df [feat_col].mean ()
            x [period_idx] = period_mean
            result [str (period_idx*mths_to_combine) + '_mths_prior'].append (period_mean)

        result ['q0'].append (x [0])
        result ['q1'].append (x [1])
        result ['q2'].append (x [2])
        result ['q3'].append (x [3])
        result ['q4'].append (x [4])

        base = None
        numer = None
        
        if mths_to_combine == 12:
            base = np.mean (x [1:])
            numer = np.mean (x [:1])
        if mths_to_combine == 6:
            base = np.mean (x [2:])
            numer = np.mean (x [:2])
            result ['past_year_1'].append (numer)
            result ['past_year_2'].append (base)
        if mths_to_combine == 3:
            base = np.nanmean (x [4:])
            numer = np.nanmean (x [:4])
            result ['past_year_1'].append (numer)
            result ['past_year_2'].append (base)
        if base != 0 and np.isfinite (base) and np.isfinite (numer):
            result ['yoy_ratio'].append (numer/base)
        else:
            result ['yoy_ratio'].append (np.nan)
        result ['yoy_diff'].append (numer - base)
        
        q_latest = x [0]
        q_prev = x [1]

        result ['past_qtr_1'].append (q_latest)
        result ['past_qtr_2'].append (q_prev)
        if q_prev != 0 and np.isfinite (q_prev):
            result ['qoq_ratio'].append (q_latest/ q_prev)
        else:
            result ['qoq_ratio'].append (np.nan)
        result ['qoq_diff'].append (q_latest - q_prev)
        
    return pd.DataFrame (result)


def main ():
    data = load_data (refresh_live_sources=False)
    input_series, label_series = add_standard_columns (data)
    features_df = get_featurized_inputs (input_series, label_series, mths_to_combine=12)
    import pdb; pdb.set_trace ()
    zzz = 1


if __name__ == "__main__":
    main()

