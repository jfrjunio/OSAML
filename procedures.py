from sklearn.model_selection import learning_curve, train_test_split
from sklearn.feature_selection import SelectFromModel
from joblib import Parallel, delayed
from matplotlib import colors
from sklearn.metrics import * 

import os
import time
import warnings
import itertools
import scipy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import pickle as pickle
import matplotlib as mpl
import _pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', UserWarning)

def standard_figsize(nx=1, ny=1, ratio=1.3, scale=1):
    """ Returns a tuple to be used as figure size.
    Arguments:
        scale:
        nx:
        ny:
        ratio:
    Returns:
    """
    ratio = 1.61803398875 if ratio < 0 else ratio
    return 4.66 * ratio * scale * nx, 4.66 * scale * ny


def enforce_styles(use_tex=False):
    """ Set several mpl.rcParams and sns.set_style for my taste.
    Arguments:
        use_tex:
    Returns:
        None
    """
    # seaborn styles
    sns.set_style('white')
    sns.set_style({
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    # matplotlib styles
    styles = {
        'text.usetex': use_tex,
        #'font.family': 'sans-serif',
        #'font.sans-serif': ['Helvetica'],
        'text.latex.unicode': True,
        'text.latex.preamble': [
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\usepackage{amsmath}",
            r"\usepackage{mathptmx}"
        ],
        'axes.labelsize': 30,
        'axes.titlesize': 30,
        'ytick.right': 'off',
        'xtick.top': 'off',
        'ytick.left': 'on',
        'xtick.bottom': 'on',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        'axes.linewidth': 1.8,
        'xtick.major.width': 1.8,
        'xtick.minor.width': 1.8,
        'xtick.major.size': 14,
        'xtick.minor.size': 7,
        'xtick.major.pad': 10,
        'xtick.minor.pad': 10,
        'ytick.major.width': 1.8,
        'ytick.minor.width': 1.8,
        'ytick.major.size': 14,
        'ytick.minor.size': 7,
        'ytick.major.pad': 10,
        'ytick.minor.pad': 10,
        'axes.labelpad': 15,
        'axes.titlepad': 15,
        "xtick.direction": "in",
        "ytick.direction": "in"
    }
    mpl.rcParams.update(styles)


def visual_benchmark(scores, estimators, filename, color='#305d8a', figsize=(15, 8), overridefig=False,
                     palette='RdBu', savefig=True, strict_positive=False,
                     verbose=False, ylim=None, ylabel='Score'):
    fig, ax = plt.subplots(figsize=figsize)

    if strict_positive:
        pos_estimators = []
        for key in estimators:
            if min(scores[key]) > 0:
                pos_estimators.append(key)
            elif verbose:
                print('%30s will be removed from the plot because it has negative values.' % key)
        estimators = pos_estimators
    
    x_ordered = np.array(sorted([(key, np.median(scores[key])) for key in estimators], key=lambda k: k[1]))[:, 0]
    # x_ordered = np.array([(key, np.median(scores[key])) for key in estimators])[:, 0]
    
    x = [i for i in range(0, len(estimators))]
    y = [scores[key] for key in x_ordered]

    if palette is None:
        sns.boxplot(x=x, y=y, color=color)
    else:
        sns.boxplot(x=x, y=y, palette=palette)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='y', direction='out', labelsize=20)
    ax.tick_params(axis='x', direction='out', labelsize=20)

    # ax.grid(axis='y', linestyle='-', alpha=0.4)

    # reducing the size of the model (both to classifiers and regressors)
    # standardized_estimators = standardized_model_name(list(estimators))
    standardized_estimators = standardized_model_name(list(x_ordered))

    ax.set_xlim(-1, len(standardized_estimators))
    ax.set_xticklabels(standardized_estimators, rotation=45, ha='right')

    if ylim is not None:
        plt.ylim(*ylim)

    ax.set_ylabel(ylabel)
    # ax.set_xlabel(r'Machine learning algorithm')

    if savefig:
        path = 'images/%s.pdf' % filename
        if not os.path.isfile(path) or overridefig:
            fig.savefig(path, bbox_inches='tight', format='pdf')
        # path = 'images/%s.svg' % filename
        # if not os.path.isfile(path) or overridefig:
        #     fig.savefig(path, bbox_inches='tight', format='svg')
        # path = 'images/%s.png' % filename
        # if not os.path.isfile(path) or overridefig:
        #     fig.savefig(path, bbox_inches='tight', dpi=480, format='png')

    return fig, ax


def algorithm_benchmark(x, y, test_size, iterations, filename, estimators, ignored_estimators=[],
                        n_jobs=-1, override=False, verbose=False):

    def parallel_pool(x, y, estimator):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        est = estimator()
        # treating particular cases when using a single feature
        if x.shape[1] == 1 and 'n_components' in est.get_params().keys():
            est.set_params(**{'n_components': 1})
        # limiting the number of parallel processes
        if 'n_jobs' in est.get_params().keys():
            est.set_params(**{'n_jobs': 1})
        # training and scoring
        est.fit(x_train, y_train)
        return est.score(x_test, y_test)

    pkl_filename = 'data/%s.pkl' % filename
    if not os.path.isfile(pkl_filename) or override:
        scores = {}
        for name in sorted(set(estimators.keys()) - set(ignored_estimators)):
            try:
                start = time.time()
                scores[name] = np.array(
                    Parallel(n_jobs=n_jobs)(delayed(parallel_pool)(x, y, estimators[name]) for _ in range(iterations)))
                if verbose:
                    print('%30s succeed in %7.02f seconds.' % (name, time.time() - start))

            except Exception as e:
                if verbose:
                    print('%30s failed because of %s!' % (name, type(e).__name__))

        with open(pkl_filename, 'wb') as file:
            cPickle.dump(scores, file, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_filename, 'rb') as file:
            scores = cPickle.load(file)

    return scores
