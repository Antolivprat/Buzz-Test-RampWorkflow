from __future__ import division, print_function
import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.prediction_types import make_regression

from rampwf.utils.importing import import_file

problem_title = 'Predicting Buzz of a theme'

# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------

workflow = rw.workflows.FeatureExtractorRegressor()

# -------------------------------------------------------------------
# The prediction type (class) to create wrapper objects for `y_pred`,
# -------------------------------------------------------------------

Predictions = make_regression()

# -----------------------------------------------------------------------------
# Score Metrics
# -----------------------------------------------------------------------------

class R2(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='R2', precision=2):
        self.name = name
        self.precision = precision
        self.r2_score = r2_score

    def __call__(self, y_true, y_pred):
        var = self.r2_score(y_true, y_pred)
        return var
    
class MS_abs_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='MSLE', precision=2):
        self.name = name
        self.precision = precision
        self.mean_absolute_error = mean_absolute_error

    def __call__(self, y_true, y_pred):
        y_pred = np.array(list(map(lambda x: x if x > 0 else 0, y_pred))) 
        var = self.mean_absolute_error(y_true, y_pred)
        return var/len(y_true)
    
class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='RMSE_5pres', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        result = np.square(y_true - y_pred)
        less_precision = np.array(list(map(lambda x: x if x>0 else 0, result - 5**2))) 
        return np.sqrt(np.mean(less_precision))/len(y_true)

score_types = [
        #5 point precision RMSE
    RMSE(),
    #MeanSquaredLogError
    MS_abs_error(),
    #Explained Variance
    R2()
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):
    # using 5 folds as default
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------

def _read_data(path, type_):
    
    fname = 'data_{}.csv'.format(type_)
    fp = os.path.join(path, 'data', fname)
    data = pd.read_csv(fp)

    fname = 'labels_{}.csv'.format(type_)
    fp = os.path.join(path, 'data', fname)
    y = pd.read_csv(fp, squeeze = True)

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 5000
        data = data[:N_small]
        y = y[:N_small]

    return data, y


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')


