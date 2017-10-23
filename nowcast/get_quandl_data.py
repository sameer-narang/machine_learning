#!/usr/bin/python3.5

import quandl

QUANDL_DATA_CODES = {
    WTI_DeptOfEnergy: "EIA/PET_RWTC_D"
}

def get_quandl_data ():
    for dc in QUANDL_DATA_CODES:
        data = quandl.get (QUANDL_DATA_CODES [dc])
        print ("--------------------------------------------")
        print (data) 
        print ("--------------------------------------------")

