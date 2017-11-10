#!/usr/bin/python3.5

import json
import numpy as np
import pandas as pd

from datetime import datetime
from pprint import pprint

DATA_DIR = 'auto/'

def convert_to_df (fname):
    block = None
    new_acy = {'addr': [], 'time': [], 'amt': [], 'spent': [], 'seller': []}
    wallet_to_wallet = {'addr': [], 'time': [], 'amt': [], 'spent': [], 'seller': []}

    with open (fname) as data_file:
        block = json.load (data_file)

    for blk in block ['blocks']:
        for txn in blk ['tx']:
            txn_time = datetime.fromtimestamp (txn ['time'])
            if 'prev_out' in txn ['inputs'] [0]:
                for src_txn in txn ['inputs']:
                    if 'addr' not in src_txn ['prev_out']:
                        continue
                    wallet_to_wallet ['time'].append (txn_time)
                    wallet_to_wallet ['amt'].append (src_txn ['prev_out'] ['value'])
                    wallet_to_wallet ['seller'].append (True)
                    wallet_to_wallet ['spent'].append (src_txn ['prev_out'] ['spent'])
                    wallet_to_wallet ['addr'].append (src_txn ['prev_out'] ['addr'])
                for dest_txn in txn ['out']:
                    if 'addr' not in dest_txn:
                        continue
                    wallet_to_wallet ['time'].append (txn_time)
                    wallet_to_wallet ['amt'].append (dest_txn ['value'])
                    wallet_to_wallet ['seller'].append (False)
                    wallet_to_wallet ['spent'].append (dest_txn ['spent'])
                    wallet_to_wallet ['addr'].append (dest_txn ['addr'])
            else:
                for dest_txn in txn ['out']:
                    if 'addr' not in dest_txn:
                        continue
                    new_acy ['time'].append (txn_time)
                    new_acy ['amt'].append (dest_txn ['value'])
                    new_acy ['seller'].append (False)
                    new_acy ['spent'].append (dest_txn ['spent'])
                    new_acy ['addr'].append (dest_txn ['addr'])
                
    return new_acy, wallet_to_wallet

s, d = convert_to_df (DATA_DIR + '493938.json')

