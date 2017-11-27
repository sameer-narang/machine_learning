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
        "code": "BEA/T20805_M",
        "format": Constant.TS,
        "data": None,
        "oldest": "1959-01-31",
        "latest": "2017-09-30"
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
        "code": "BEA/T10107_Q",
        "format": Constant.TS,
        "data": None,
        "oldest": "1947-06-30",
        "latest": "2017-09-30"
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
    # since 1985, latest: 2017-11-17
    "ECO_UNCERTAINTY_FedReserve": {
        "refresh": True,
        "code": "FRED/USEPUINDXD",
        "format": Constant.TS,
        "data": None,
        "oldest": "1985-01-01",
        "latest": "2017-11-17"
    },
    # since 1949, latest: 2017-10-01
    "LONG_TERM_UNEMPL_FedReserve": {
        "refresh": True,
        "code": "FRED/NROU",
        "format": Constant.TS,
        "data": None,
        "oldest": "1949-01-01",
        "latest": "2017-10-01"
    },
    "FIXED_CAP_CONS_Bea": {
        "refresh": True,
        "code": "BEA/T70500_Q",
        "format": Constant.TS,
        "data": None,
        "oldest": "1947-03-31",
        "latest": "2017-09-30"
    },
    "TEXTILE_INVENTORIES_FedReserve": {
        "refresh": True,
        "code": "FRED/U14STI",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-09-01"
    },
    "DURABLE_INVENTORIES_FedReserve": {
        "refresh": True,
        "code": "FRED/AODGTI",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-10-01"
    },
    "DURABLE_SHIPMENTS_FedReserve": {
        "refresh": True,
        "code": "FRED/UODGVS",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-10-01"
    },
    "TEXTILE_SHIPMENTS_FedReserve": {
        "refresh": True,
        "code": "FRED/U14SVS",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-09-01"
    }
    ,"RES_BLDG_PERMITS_FedReserve": {
        "refresh": True,
        "code": "FRED/PERMITNSA",
        "format": Constant.TS,
        "data": None,
        "oldest": "1959-01-01",
        "latest": "2017-10-01"
    }
    ,"NE_HOUSING_STARTS_FedReserve": {
        "refresh": True,
        "code": "FRED/HOUST2UMNEQ",
        "format": Constant.TS,
        "data": None,
        "oldest": "1985-01-01",
        "latest": "2017-07-01"
    }
    ,"SOUTH_HOUSING_STARTS_FedReserve": {
        "refresh": True,
        "code": "FRED/HOUST2UMSQ",
        "format": Constant.TS,
        "data": None,
        "oldest": "1985-01-01",
        "latest": "2017-07-01"
    }
    ,"WEST_HOUSING_STARTS_FedReserve": {
        "refresh": True,
        "code": "FRED/HOUST2UMWQ",
        "format": Constant.TS,
        "data": None,
        "oldest": "1985-01-01",
        "latest": "2017-07-01"
    }
    ,"CPCTY_UTIL_FedReserve": {
        "refresh": True,
        "code": "FRED/TCU",
        "format": Constant.TS,
        "data": None,
        "oldest": "1967-01-01",
        "latest": "2017-10-01"
    }
    ,"IND_PROD_FedEco": {
        "refresh": True,
        "code": "FED/IP_B50001_N",
        "format": Constant.TS,
        "data": None,
        "oldest": "1919-01-31",
        "latest": "2017-10-31"
    }
    ,"GEN_BIZ_NY_FederalReserve": {
        "refresh": True,
        "code": "FRED/GACNNA156MNFRBNY",
        "format": Constant.TS,
        "data": None,
        "oldest": "1919-01-31",
        "latest": "2017-10-31"
    }
    ,"RETAIL_SALES_FederalReserve": {
        "refresh": True,
        "code": "FRED/RSAFSNA",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-10-01"
    }
    ,"PPI_ALL_COMMODITIES_FederalReserve": {
        "refresh": True,
        "code": "FRED/PPIACO",
        "format": Constant.TS,
        "data": None,
        "oldest": "1913-01-01",
        "latest": "2017-10-01"
    }
    ,"JOB_OPENINGS_FederalReserve": {
        "refresh": True,
        "code": "FRED/JTSJOL",
        "format": Constant.TS,
        "data": None,
        "oldest": "2012-12-01",
        "latest": "2017-09-01"
    }
    ,"NMI_ISM": {
        "refresh": True,
        "code": "ISM/NONMAN_NMI",
        "format": Constant.TS,
        "data": None,
        "oldest": "2008-01-01",
        "latest": "2017-10-01"
    }
    ,"TTL_INVENTORIES_FederalReserve": {
        "refresh": True,
        "code": "FRED/UMTMTI",
        "format": Constant.TS,
        "data": None,
        "oldest": "1992-01-01",
        "latest": "2017-09-01"
    }
    ,"IMPORTS_GOODS_N_SERVICES_FederalReserve": {
        "refresh": True,
        "code": "FRED/IEAMGSN",
        "format": Constant.TS,
        "data": None,
        "oldest": "1999-01-01",
        "latest": "2017-04-01"
    }
    ,"EXPORTS_GOODS_N_SERVICES_FederalReserve": {
        "refresh": True,
        "code": "FRED/EXPGS",
        "format": Constant.TS,
        "data": None,
        "oldest": "1947-01-01",
        "latest": "2017-07-01"
    }
}

def get_quandl_data (code=None):
    for dc in QUANDL_DATA:
        if code is None:
            if not QUANDL_DATA [dc]["refresh"]:
                continue
        elif dc not in code:
            continue
        else:
            None
            
        ### sn: to get a numpy array, use the below. Don't worry about optimizing at this pt.
        ### quandl.get (QUANDL_DATA [dc] ["code"], returns = "numpy")
        
        print ("Fetching data using the code: " + dc)

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

