import copy 

OBJ_FN_PCT_THRESHOLD = 0.05

def scale_and_add_sparse_vectors (scale1, d1, scale2=1, d2=None):
    """
    d1 is modified in place to scale by scale1, and if d2 is supplied, it is added to d1 after scaling by scale2
    """
    for k, v in d1.items ():
        d1 [k] = scale1 * v
        
    if not d2:
        return d1
    
    for k, v in d2.items ():
        if k in d1:
            d1 [k] += scale2 * v
        else:
            d1 [k] = scale2 * v


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


class PegasosClassifier (object):
    
    def __init__ (self, lambda_val):
        self._lambda = lambda_val
    
    def _evaluate_obj_function (self, X, y):
        w_sqr = dotProduct (self._w, self._w)
        reg_term = 0.5 * self._lambda * w_sqr
        i = 0
        correct_conf_predictions = 0
        while i < len (y):
            if y [i] * dotProduct (self._w, X[i]) > 1:
                correct_conf_predictions += 1
            i = i + 1
  
        pct_incorrect_predictions = (1 - correct_conf_predictions/ len (y))
        an_val = reg_term + pct_incorrect_predictions

        print ("obj_fn_val: %.3f, reg. term: %.3f, pct_incorrect_predictions: %.3f" % \
               (fn_val, reg_term, pct_incorrect_predictions))
        return fn_val, correct_conf_predictions
        
    def train_model (self, X, y, w={}):
        iterate = True
        self._w = copy.deepcopy (w)
        t = 0
        obj_fn_prev = 1e9
        while iterate:
            w_prev = copy.deepcopy (self._w)
            idx = 0
            while idx < len (y):
                t = t + 1
                eta = 1/ (self._lambda * t)
                if y[idx] * dotProduct (X [idx], self._w) < 1:
                    scale_and_add_sparse_vectors ((1 - eta * self._lambda), self._w, (eta * y [idx]), X [idx])
                else:
                    scale_and_add_sparse_vectors ((1 - eta * self._lambda), self._w)
                idx += 1

            obj_fn_cur, correct_predictions_cur = self._evaluate_obj_function (X, y)

            if obj_fn_cur >= obj_fn_prev:
                iterate = False
                self._w = w_prev
            elif obj_fn_cur * (1 + OBJ_FN_PCT_THRESHOLD) > obj_fn_prev:
                iterate = False
            else:
                iterate = True
                obj_fn_prev = obj_fn_cur

lambda_val = 0.1
classifier = PegasosClassifier (lambda_val)
classifier.train_model (X_train, y_train)

