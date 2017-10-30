import collections
import copy
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split

POS_DATA_DIR="./data/pos"
NEG_DATA_DIR="./data/neg"
SEQUENCES_TO_IGNORE = set ([',', '.', "'", '"', ';', ':'])
SEED = 5
OBJ_FN_PCT_THRESHOLD = 0.05

# this function returns 2 lists - X, y where each entry of X is a string that has the whole text
# of a review, and the corresponding element of y is +1 for a positive review, -1 for a negative review
def read_raw_data (data_dir, yval):
    X = []
    files = [os.path.join(data_dir, f) for f in os.listdir (data_dir) if os.path.isfile (os.path.join (data_dir, f))]
    for f in files:
        with open (f, 'r') as x:
            X.append (x.read ().replace ('\n', ''))
    
    y = [yval] * len (files)
    return X, y
    
def convert_str_to_bag_of_words (input_str):
    words = set (input_str.split ())
    relevant_words = words - SEQUENCES_TO_IGNORE
    cntr = collections.Counter (relevant_words)
    return cntr

def get_full_dataset ():
    X = []
    y = []

    X_raw, y_partial = read_raw_data (POS_DATA_DIR, 1)
    for x in X_raw:
        X.append (convert_str_to_bag_of_words (x))
    y.extend (y_partial)
    
    X_raw, y = read_raw_data (NEG_DATA_DIR, -1)
    for x in X_raw:
        X.append (convert_str_to_bag_of_words (x))
    y.extend (y_partial)

    return X, y

def split_data (X, y):
    X_train, X_val_and_test, y_train, y_val_and_test = \
            train_test_split (X, y, test_size=0.25, random_state=SEED)
    X_val, X_test, y_val, y_test = \
            train_test_split (X_val_and_test, y_val_and_test, test_size=0.0, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test

# sn: make calls to functions defined above
X, y = get_full_dataset ()
X_train, X_val, X_test, y_train, y_val, y_test = split_data (X, y)
