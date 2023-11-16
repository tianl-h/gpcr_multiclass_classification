# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2020-12-15 13:50:32
# @Last Modified by:   tianl
# @Last Modified time: 2021-01-11 14:31:36

from utils.utils import create_labels
from utils.utils import get_model
from utils.models import METHODS_WITH_HYPER
from utils.feature_reduction import feature_reduction

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC


import numpy as np
import matplotlib.pyplot as plt

import json
import os
import time

def train(model, X_train_prime, y_train_prime, K=5, hyper=None):
    clf = GridSearchCV(model, hyper, cv=K)
    clf.fit(X_train_prime, y_train_prime)
    sorted(clf.cv_results_.keys())
    return clf

def main():
    data_fn = 'data/GPCR.npz'
    stats_fn = 'data/stats.json'
    K = 5 # KFold CV
    test_size = 0.2
    random_state = 2020 # for train_test_split

    # import data
    data_by_atts = np.load(data_fn)
    with open(stats_fn, 'r') as f:
        stats = json.load(f)

    for att, v in stats.items():
        stats[att] = v['Atompair']

    data_all_features = np.concatenate((
                        data_by_atts['Atompair'],
                        data_by_atts['ECFP6'],
                        data_by_atts['MACCS'],
                        data_by_atts['PC']), axis = 1)

    data = {'Atompair': data_by_atts['Atompair'], 'ECFP6': data_by_atts['ECFP6'], 'MACCS': data_by_atts['MACCS'], 'PC': data_by_atts['PC']}

    data_atom_pc = np.concatenate((
                data_by_atts['Atompair'],
                data_by_atts['PC']), axis = 1)

    # create labels
    labels, names_by_labels = create_labels(stats)
    X_train_prime, X_test, y_train_prime, y_test = train_test_split(data_atom_pc, labels,
                                                test_size=test_size, random_state=random_state, stratify=labels)

    ### Test HyperParam
    start = time.time()
    clf = SVC(C=10)
    clf.fit(X_train_prime, y_train_prime)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    stop = time.time()
    print("Training time: %.3f mins" % ((stop - start) / 60))
    print('Accuracy: %.3f' % (acc))

    for att, d in data.items():
        X_train_prime, X_test, y_train_prime, y_test = train_test_split(d, labels,
                                                test_size=test_size, random_state=random_state, stratify=labels)
        for method, func_and_hyper in METHODS_WITH_HYPER.items():
            start = time.time()
            func = func_and_hyper['func']
            hyper = func_and_hyper['hyper']

            model = get_model(method, func)
            clf = train(model, X_train_prime, y_train_prime, K=K, hyper=hyper)
            res = clf.cv_results_
            best_idx = np.argmax(res['mean_test_score'])
            best_param = res['params'][best_idx]
            clf = get_model(method, func, param=best_param)
            clf.fit(X_train_prime, y_train_prime)
            filename = att + '_' + method + '.sav'
            pickle.dump(clf, open(filename, 'wb'))
            loaded_classifier = pickle.load(open(filename, 'rb'))
            result = loaded_classifier.score(x_test, y_test)
            print(result)

            y_test_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)

            print('%6s best hyper parameter: %30s, Accuracy: %.3f' % (method.upper(), str(best_param), acc))
            stop = time.time()
            print("Training time: %.3f mins" % ((stop - start) / 60))

if __name__ == '__main__':
    main()
