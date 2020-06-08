from xgboost import XGBRegressor
from sklearn.utils.testing import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
import sklearn

import os
import warnings
import scipy as sc
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import pickle as pickle
import _pickle as cPickle
import matplotlib.pyplot as plt
import sys

from .procedures import *

# Defining Matplolib and Seaborn styles.
enforce_styles(use_tex=False)

# Listing all regressors in the sklearn library, which are an instance of RegressorMixin.
Regression = False
if Regression:
    algorithm = {est[0]: est[1] for est in all_estimators() if issubclass(est[1], RegressorMixin)}
    for alg in algorithm:
        print(alg)
else:
    # Listing all classifiers in the sklearn library, which are an instance of
    algorithm = {est[0]: est[1] for est in all_estimators() if issubclass(est[1], ClassifierMixin)}
    for alg in algorithm:
        print(alg)

# Defining the default figure ratio.
xs, ys = standard_figsize()

# The number of cores used to run the multi-thread algorithms.
n_jobs = -1 # whenever set to -1 it uses all the cores regardless of being available or not

# Appending sklearn-friendly third-party regressors.
algorithm['XGBRegressor'] = XGBRegressor

df = pd.read_csv('01.PATIENTS_VISITES-no_control_no_spurious_data-median-imputation_ADASYN.csv',sep=',')
df = sklearn.utils.shuffle(df)

X = df[['age', 'egfdrs001', 'nycturie_nb', 'score_depression', 'imc', 'perimetre_cervical', 'tour_de_hanches', 'padiast']]

if Regression:
    Y = df[['polysomnographie_iah']]
else:
    Y = df[['iah_class']]
Y = Y.values.ravel()

# Here we will test each regressor in n iterations considering m cross-validation folds splitting the dataset randomly according to test_size.

reg_score = algorithm_benchmark(x=X, y=Y, test_size=0.33,
                                iterations=300, filename='%sREGs', estimators=algorithm,
                                ignored_estimators=ignored_reg, n_jobs=n_jobs, override=True, verbose=True)

# Visualizing the results of all regressors.
if Regression:
    fig = visual_benchmark(scores=reg_score, estimators=reg_score.keys(), filename='AllRegressorsR2Score',
                           color='#6d6d6d', figsize=(xs*2, ys), overridefig=True, palette='viridis',
                           savefig=True, strict_positive=True, verbose=True, ylim=(-1, 1), ylabel='$R^2$-score')
else:
    fig = visual_benchmark(scores=reg_score, estimators=reg_score.keys(), filename='AllClassifiersAccuracy',
                           color='#6d6d6d', figsize=(xs*2, ys), overridefig=True, palette='viridis',
                           savefig=True, strict_positive=False, verbose=True, ylim=(-1, 1), ylabel='Accuracy')
