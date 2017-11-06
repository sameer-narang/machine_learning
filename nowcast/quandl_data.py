#!/usr/bin/python3.5

import pandas as pd
import quandl

import util

from util import *

quandl.ApiConfig.api_key = "8DrzzNxwBzDzh_1jE6Dj"

QUANDL_DATA = {
    # since 1969, latest: 2017-08-31
    "PCE_Bea": {
        "refresh": True,
        "code": "BEA/NIPA_2_8_5_M",
        "format": Constant.TS,
        "data": None,
        "oldest": "1969-01-31",
        "latest": "2017-08-31"
    },
    "WTI_DeptOfEnergy": {
        "refresh": True,
        "code": "EIA/PET_RWTC_D",
        "format": Constant.TS,
        "data": None,
        "oldest": "1986-01-02",
        "latest": "2017-10-23"
    },
    "UNEMPLOYMENT_FedReserve": {
        "refresh": True,
        "code": "FRED/LNU03000000",
        "format": Constant.TS,
        "data": None,
        "oldest": "1948-01-01",
        "latest": "2017-09-01"
    },
    # since 1994
    "PREM_GAS_PX_FedReserve": {
        "refresh": True,
        "code": "FRED/GASPRMCOVM",
        "format": Constant.TS,
        "data": None,
        "oldest": "1994-12-01",
        "latest": "2017-10-01"
    },
    # since 1969, latest: 2017-06-30
    "QTRLY_GDP_PCT_CHG_Bea": {
        "refresh": True,
        "code": "BEA/NIPA_1_1_1_Q",
        "format": Constant.TS,
        "data": None,
        "oldest": "1969-03-31",
        "latest": "2017-06-30"
    },
    # since 1971, latest: 2017-10-26
    "AVG_30Y_MTGG_FreddieMac": {
        "refresh": True,
        "code": "FMAC/30US",
        "format": Constant.TS,
        "data": None,
        "oldest": "1971-04-02",
        "latest": "2017-10-26"
    },
    # since 1890, latest: 2017-12-31
    "REAL_BLDG_COST_Yale": {
        "refresh": True,
        "code": "YALE/RBCI",
        "format": Constant.TS,
        "data": None,
        "oldest": "1890-12-31",
        "latest": "2015-12-31"
    },
    # since 1871, latest: 2017-10-31
    "S&P_COMP_Yale": {
        "refresh": True,
        "code": "YALE/SPCOMP",
        "format": Constant.TS,
        "data": None,
        "oldest": "1871-01-31",
        "latest": "2017-10-31"
    },
    # since 1989, latest: 2017-09-30
    "INDIV_STK_MKT_CONF_Yale": {
        "refresh": True,
        "code": "YALE/US_CONF_INDEX_VAL_INDIV",
        "format": Constant.TS,
        "data": None,
        "oldest": "1989-10-31",
        "latest": "2017-09-30"
    },
    # since 1989, latest: 2017-09-30
    "INSTI_STK_MKT_CONF_Yale": {
        "refresh": True,
        "code": "YALE/US_CONF_INDEX_VAL_INST",
        "format": Constant.TS,
        "data": None,
        "oldest": "1989-10-31",
        "latest": "2017-09-30"
    },
}

def get_quandl_data ():
    for dc in QUANDL_DATA:
        if not QUANDL_DATA [dc]["refresh"]:
            continue
            
        ### sn: to get a numpy array, use the below. Don't worry about optimizing at this pt.
        ### quandl.get (QUANDL_DATA [dc] ["code"], returns = "numpy")
        
        if QUANDL_DATA [dc]["format"] == Constant.TS:
            QUANDL_DATA [dc] ["data"] = quandl.get (QUANDL_DATA [dc] ["code"])
            save_quandl_data_to_file (dc)
        elif QUANDL_DATA [dc]["format"] == Constant.DT:
            QUANDL_DATA [dc] ["data"] = quandl.get_table (QUANDL_DATA [dc] ["source"], ticker=QUANDL_DATA [dc] ["ticker"])
            save_quandl_data_to_file (dc)
        else:
            print ("Error - unrecognized format specified in Quandl data source - " + dc)
            None

def save_quandl_data_to_file (dc):
    df = QUANDL_DATA [dc] ["data"]
    if isinstance (df, pd.DataFrame):
        file_name = Constant.DATA_DIR + "/" + dc + ".csv"
        print ("Saving dataframe to file " + file_name)
        df.to_csv (file_name, encoding='utf-8')

