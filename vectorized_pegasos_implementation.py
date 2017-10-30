import copy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# sn: this function returns arrays of strings/ text for use by CountVectorizer
def get_full_dataset_as_text ():
    X_raw = []
    y = []

    X_raw_partial, y_partial = read_raw_data (POS_DATA_DIR, 1)
    X_raw.extend (X_raw_partial)
    y.extend (y_partial)
    
    X_raw, y = read_raw_data (NEG_DATA_DIR, -1)
    X_raw.extend (X_raw_partial)    
    y.extend (y_partial)

    return X_raw, y

class VectorizedPegasosClassifier (object):
    
    def __init__ (self, lambda_val):
        self._lambda = lambda_val

    def _evaluate_obj_function (self, X, y):
        w_sqr = (self._s ** 2) * np.dot (self._w, self._w)
        reg_term = 0.5 * self._lambda * w_sqr
        i = 0
        correct_conf_predictions = 0
        while i < len (y):
            if y [i] * np.dot (self._w, X[i]) > 1:
                correct_conf_predictions += 1
            i = i + 1
  
        '''
        for feature in self._w:
            if (self._w [feature] > ABS_WT_THRESHOLD):
                print (feature + ": " + str (self._w [feature]))
        '''
        pct_incorrect_predictions = (1 - correct_conf_predictions/ len (y))
        fn_val = reg_term + pct_incorrect_predictions

        print ("obj_fn_val: %.3f, reg. term: %.3f, pct_incorrect_predictions: %.3f" % \
               (fn_val, reg_term, pct_incorrect_predictions))
        return fn_val, pct_incorrect_predictions
        
    def train_model (self, X, y):
        # X is received as a bag-of-words dictionary
        iterate = True
        self._w = np.zeros (X.shape [1])
        self._s = 0
        t = 0
        obj_fn_prev = 1e9
        pct_errors_prev = 1
        while iterate:
            w_prev = copy.deepcopy (self._w)
            s_prev = self._s
            idx = 0
            while idx < len (y):
                t = t + 1
                eta = 1/ (self._lambda * t)
                self._s = (1 - eta * self._lambda) * self._s
                if self._s == 0:
                    self._s = 1
                    self._w = self._w = np.zeros (X.shape [1]) 
                if y[idx] * np.dot (X [idx], self._w) < 1:
                    self._w = self._w + eta * y [idx] * X [idx]/ self._s

                idx += 1

            obj_fn_cur, pct_errors = self._evaluate_obj_function (X, y)

            if obj_fn_cur >= obj_fn_prev:
                iterate = False
                self._w = w_prev
                pct_errors = pct_errors_prev
                obj_fn_cur = obj_fn_prev
            elif obj_fn_cur * (1 + OBJ_FN_PCT_THRESHOLD) > obj_fn_prev:
                iterate = False
            else:
                iterate = True
                obj_fn_prev = obj_fn_cur
                pct_errors_prev = pct_errors
                
        return obj_fn_cur, pct_errors
    
X, y = get_full_dataset_as_text ()
X_train, X_val, X_test, y_train, y_val, y_test = split_data (X, y)
vectorizer = CountVectorizer ()
X_train_vectors = vectorizer.fit_transform (X_train).toarray ()
X_val_vectors = vectorizer.transform (X_val).toarray()
    
classifier2 = VectorizedPegasosClassifier(0.1)
classifier2.train_model (X_train_vectors, y_train)
