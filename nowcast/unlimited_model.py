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

def rf_featurize_series (days_to_qtr_end, x_df, y_df, num_days_per_period=91, num_years=2, fill_na_with=0):
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
    #x_df ['Total'] = x_df ['Total'].fillna (fill_na_with)

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
            period_mean = tmp_df ['Total'].mean ()
            if not np.isnan (period_mean):
                x [period_idx] = tmp_df ['Total'].mean ()
       
        X.append (x)
        quarters.append (qtr_end)
        y.append (row ['Gross domestic product'])

    # import pdb; pdb.set_trace ()
    # sn: formatted print - keep this
    # print ('\n'.join ([', '.join ("{0:.1f}".format(d) for d in row) for row in X]))
    return np.array (X), np.array (y)


# sn: this function returns X, y - the full data set since the beginning of y
# each day in the prior 2 years is a feature
# the function takes as argument the number of days prior to quarter end that we are
# predicting the GDP growth for
# x_df and y_df must have the column 'Date' and x_df must also have the column 'Total'
'''
def featurize_series_old (days_to_qtr_end, x_df, y_df, num_days_per_period=91, num_years=2):
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
'''

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
                if series_name == 'gdp':
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
        y_preds = self._model.predict (self._X_val)

        self.print_summary (y_preds, self._y_val)

    def print_test_data_performance (self):
        y_preds = None
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

    pce_df = QUANDL_DATA ["PCE_Bea"] ["data"]
    wti_df = QUANDL_DATA ["WTI_DeptOfEnergy"] ["data"]
    unempl_df = QUANDL_DATA ["UNEMPLOYMENT_FedReserve"] ["data"]
    avg30ymtg_df = QUANDL_DATA ["AVG_30Y_MTGG_FreddieMac"] ["data"]
    bldg_cost_df = QUANDL_DATA ["REAL_BLDG_COST_Yale"] ["data"]
    snp_df = QUANDL_DATA ["S&P_COMP_Yale"] ["data"]
    indiv_conf_df = QUANDL_DATA ["INDIV_STK_MKT_CONF_Yale"] ["data"]
    insti_conf_df = QUANDL_DATA ["INSTI_STK_MKT_CONF_Yale"] ["data"]
    act_gdp_df = QUANDL_DATA ["QTRLY_GDP_PCT_CHG_Bea"] ["data"]
    eco_uncertainty_df = QUANDL_DATA ["ECO_UNCERTAINTY_FedReserve"] ["data"]
    lt_unempl_df = QUANDL_DATA ["LONG_TERM_UNEMPL_FedReserve"] ["data"]
    snp_earnings_df = QUANDL_DATA ["S&P_COMP_Yale"] ["data"]
    fixed_cap_consumption_df = QUANDL_DATA ["FIXED_CAP_CONS_Bea"] ["data"]
    textile_inventories_df = QUANDL_DATA ["TEXTILE_INVENTORIES_FedReserve"] ["data"]
    textile_shipments_df = QUANDL_DATA ["TEXTILE_SHIPMENTS_FedReserve"] ["data"]
    durable_inventories_df = QUANDL_DATA ["DURABLE_INVENTORIES_FedReserve"] ["data"]
    durable_shipments_df = QUANDL_DATA ["DURABLE_SHIPMENTS_FedReserve"] ["data"]
    res_bldg_permits_df = QUANDL_DATA ["RES_BLDG_PERMITS_FedReserve"] ["data"]
    ne_housing_starts_df = QUANDL_DATA ["NE_HOUSING_STARTS_FedReserve"] ["data"]
    south_housing_starts_df = QUANDL_DATA ["SOUTH_HOUSING_STARTS_FedReserve"] ["data"]
    west_housing_starts_df = QUANDL_DATA ["WEST_HOUSING_STARTS_FedReserve"] ["data"]
    cpcty_util_df = QUANDL_DATA ["CPCTY_UTIL_FedReserve"] ["data"]
    ind_prod_df = QUANDL_DATA ["IND_PROD_FedEco"] ["data"]
    gen_biz_ny_df = QUANDL_DATA ["GEN_BIZ_NY_FederalReserve"] ["data"]

    pce_df ['Date'] = pce_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    wti_df ['Date'] = wti_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    unempl_df ['Date'] = unempl_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    avg30ymtg_df ['Date'] = avg30ymtg_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    bldg_cost_df ['Date'] = bldg_cost_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    snp_df ['Date'] = snp_df ['Year'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    indiv_conf_df ['Date'] = indiv_conf_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    insti_conf_df ['Date'] = insti_conf_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    eco_uncertainty_df ['Date'] = eco_uncertainty_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    lt_unempl_df ['Date'] = lt_unempl_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,)) 
    lt_unempl_df ['Date'] = lt_unempl_df ['Date'].apply (lambda x: x + relativedelta (years=-10))
    snp_earnings_df ['Date'] = snp_df ['Date']
    fixed_cap_consumption_df ['Date'] = fixed_cap_consumption_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    textile_inventories_df ['Date'] = textile_inventories_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    textile_shipments_df ['Date'] = textile_shipments_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    durable_inventories_df ['Date'] = durable_inventories_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    durable_shipments_df ['Date'] = durable_shipments_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    res_bldg_permits_df ['Date'] = res_bldg_permits_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    ne_housing_starts_df ['Date'] = ne_housing_starts_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    south_housing_starts_df ['Date'] = south_housing_starts_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    west_housing_starts_df ['Date'] = west_housing_starts_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    cpcty_util_df ['Date'] = cpcty_util_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    ind_prod_df ['Date'] = ind_prod_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
    gen_biz_ny_df ['Date'] = gen_biz_ny_df ['Date'].apply (datetime.datetime.strptime, args=(Constant.DATE_STR_FMT_1,))
   
    pce_df ['Total'] = pce_df ['Personal consumption expenditures (PCE)'] 
    wti_df ['Total'] = wti_df ['Value']
    unempl_df ['Total'] = unempl_df ['Value']
    avg30ymtg_df ['Total'] = avg30ymtg_df ['Value']
    bldg_cost_df ['Total'] = bldg_cost_df ['Cost Index']
    snp_df ['Total'] = snp_df ['S&P Composite']
    indiv_conf_df ['Total'] = indiv_conf_df ['Index Value']
    insti_conf_df ['Total'] = insti_conf_df ['Index Value']
    eco_uncertainty_df ['Total'] = eco_uncertainty_df ['Value']
    lt_unempl_df ['Total'] = lt_unempl_df ['Value']
    snp_earnings_df ['Total'] = snp_earnings_df ['Real Earnings']
    fixed_cap_consumption_df ['Total'] = fixed_cap_consumption_df ['Consumption of fixed capital']
    textile_inventories_df ['Total'] = textile_inventories_df ['Value']
    textile_shipments_df ['Total'] = textile_shipments_df ['Value']
    durable_inventories_df ['Total'] = durable_inventories_df ['Value']
    durable_shipments_df ['Total'] = durable_shipments_df ['Value']
    res_bldg_permits_df ['Total'] = res_bldg_permits_df ['Value']
    ne_housing_starts_df ['Total'] = ne_housing_starts_df ['Value']
    south_housing_starts_df ['Total'] = south_housing_starts_df ['Value']
    west_housing_starts_df ['Total'] = west_housing_starts_df ['Value']
    cpcty_util_df ['Total'] = cpcty_util_df ['Value']
    ind_prod_df ['Total'] = ind_prod_df ['Value']
    gen_biz_ny_df ['Total'] = gen_biz_ny_df ['Value']

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
        ,'bldg_cost': bldg_cost_df
        ,'snp': snp_df
        ,'gdp': act_gdp_df
        ,'eco_ucty': eco_uncertainty_df
        ,'lt_unempl': lt_unempl_df
        ,'indiv_conf': indiv_conf_df
        ,'insti_conf': insti_conf_df
        ,'snp_earnings': snp_earnings_df
        ,'fixed_cap_consumption': fixed_cap_consumption_df
        ,'textile_inventories': textile_inventories_df
        ,'textile_shipments': textile_shipments_df
        ,'durable_inventories': durable_inventories_df
        ,'durable_shipments': durable_shipments_df
        ,'res_bldg_permits': res_bldg_permits_df
        ,'ne_housing_starts': ne_housing_starts_df
        ,'south_housing_starts': south_housing_starts_df
        ,'west_housing_starts': west_housing_starts_df
        ,'cpcty_util': cpcty_util_df
        ,'ind_prod': ind_prod_df
        ,'gen_biz_ny': gen_biz_ny_df
    }
    label_series = act_gdp_df [act_gdp_df ['Date'] >= cut_off_date]

    lrm = Model (num_days_per_period=91, days_in_advance=0, use_scaling=False, seed=RNDM_SEED_1)
    lrm.prepare_series (input_series, label_series)

    '''
    for lambda_val in [1e-02, 1e-01, 1, 5, 1e01, 1e100]:
        print ("..................... Results with lambda=" + str (lambda_val) + ".....................")
        lrm.fit_and_summarize_ridge_model (lambda_val, use_scaling=True)
        lrm.print_test_data_performance ()
    '''
    lrm.fit_and_summarize_random_forest_model ()
    lrm.print_test_data_performance ()

    # featurize and normalize the data
    X = None
    y = None


if __name__ == "__main__":
    main()
