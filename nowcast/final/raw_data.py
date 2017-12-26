#!/usr/bin/python3.5

import os
import pandas as pd
import quandl

import util

from util import *

quandl.ApiConfig.api_key = "8DrzzNxwBzDzh_1jE6Dj"

manual_csv_codes = ['VIXCLS']

quandl_codes = set (['FED/FR515035023_Q', 'FED/FR515035023_Q', 'FED/B1027NFRDM', 'FED/B1027NCBAM',
                    'FED/B1027NCBDM', 'FED/B1027NFRAM', 'FED/FU643065105_Q', 'FED/RIW_N321991_S',
                    'FED/IP_N321991_S_Q', 'FED/FR643065173_Q', 'FED/FL643065105_Q', 'FED/FL515035023_Q',
                    'FED/IP_N321991_S', 'FED/FU515035023_Q', 'FED/IP_N321991_N_Q', 'FED/FR643065183_Q',
                    'FED/FA643065105_Q', 'FED/FA515035023_Q', 'FED/IP_N321991_N', 'FED/B1027NFRD',
                    'FED/B1027NDMDM', 'FED/B1027NCBA', 'FED/B1027NCBD', 'FED/B1027NFRA',
                    'FED/B1027NDMAM', 'FED/FU106110405_Q', 'FED/FA106110405_Q', 'FED/FU106006065_Q',
                    'FED/FA546006063_Q', 'FED/FA893092275_Q', 'FED/FU893092275_Q', 'FED/FU546006063_Q',
                    'FED/FU106110115_Q', 'FED/FU586006065_Q', 'FED/FA106110115_Q', 'FED/RESPPLLOP_N_WW',
                    'FED/FA666006305_Q', 'FED/FA546000105_Q', 'CHRIS/CBOE_VX1', "BEA/T20805_M", 
                    "EIA/PET_RWTC_D", "FRED/LNU03000000", "FRED/GASPRMCOVM", "BEA/T10107_Q",
                    "FMAC/30US", "YALE/RBCI", "YALE/US_CONF_INDEX_VAL_INDIV", "YALE/US_CONF_INDEX_VAL_INST",
                    "FRED/USEPUINDXD", "FRED/NROU", "BEA/T70500_Q", "FRED/U14STI", 
                    "FRED/AODGTI", "FRED/UODGVS", "FRED/U14SVS", "FRED/PERMITNSA",
                    "FRED/HOUST2UMNEQ", "FRED/HOUST2UMSQ", "FRED/HOUST2UMWQ", "FRED/TCU",
                    "FED/IP_B50001_N", "FRED/GACNNA156MNFRBNY", "FRED/RSAFSNA", "FRED/PPIACO",
                    "FRED/JTSJOL", "ISM/NONMAN_NMI", "FRED/UMTMTI", "FRED/IEAMGSN",
                    "FRED/EXPGS", "FED/FU145020005_Q"
])


def load_quandl_data (RAW_DATA, refresh_live_data=False):
    for code in quandl_codes:
        filecode = code.replace ('/', '_')
        RAW_DATA [filecode] = {
            "refresh": refresh_live_data,
            "code": code,
            "format": Constant.TS,
            "data" : None
        }
        if not refresh_live_data:
            file_name = Constant.DATA_DIR + "/" + filecode + ".csv"
            if not os.path.isfile (file_name):
                RAW_DATA [filecode] ['refresh'] = True
            else:
                RAW_DATA [filecode] ['data'] = pd.read_csv (file_name)
    download_quandl_data (RAW_DATA)
    print ('Loaded ' + str (len (RAW_DATA)) + ' quandl series')


def load_offline_data (RAW_DATA):
    for code in manual_csv_codes:
        RAW_DATA [code] = {
            "refresh": False,
            "code": code,
            "format": Constant.TS,
            "data" : pd.read_csv (Constant.DATA_DIR + "/" + code + ".csv")
        }
    print ('Loaded ' + str (len (RAW_DATA)) + ' manual series')
    return RAW_DATA


def download_quandl_data (QUANDL_DATA):
    for dc in QUANDL_DATA:
        if not QUANDL_DATA [dc] ["refresh"]:
                continue

        print ("Fetching quandl data using the code: " + dc)

        if QUANDL_DATA [dc] ["format"] == Constant.TS:
            QUANDL_DATA [dc] ["data"] = quandl.get (QUANDL_DATA [dc] ["code"])
            save_quandl_data_to_file (QUANDL_DATA, dc)
        elif QUANDL_DATA [dc]["format"] == Constant.DT:
            QUANDL_DATA [dc] ["data"] = quandl.get_table (QUANDL_DATA [dc] ["source"], ticker=QUANDL_DATA [dc] ["ticker"])
            save_quandl_data_to_file (QUANDL_DATA, dc)
        else:
            print ("Error - unrecognized format specified in Quandl data source - " + dc)
            None


def save_quandl_data_to_file (QUANDL_DATA, dc):
    df = QUANDL_DATA [dc] ["data"]
    if isinstance (df, pd.DataFrame):
        file_name = Constant.DATA_DIR + "/" + dc + ".csv"
        print ("Saving dataframe to file " + file_name)
        df.to_csv (file_name, encoding='utf-8')


def load_data (refresh_live_sources=False):
    RAW_DATA = {}
    load_quandl_data (RAW_DATA, refresh_live_sources)
    load_offline_data (RAW_DATA)
    return RAW_DATA

def main ():
    data = load_data (refresh_live_sources=False)
    import pdb; pdb.set_trace ()
    zzz = 1


if __name__ == "__main__":
    main()

