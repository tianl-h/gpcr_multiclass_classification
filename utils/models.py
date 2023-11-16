# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2020-12-16 14:56:32
# @Last Modified by:   tianl
# @Last Modified time: 2021-03-05 12:42:03

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

METHODS_WITH_HYPER = \
        {   
            'SVM': {'func': SVC, 'hyper':{'probability':[True],'C':[10], 'decision_function_shape': ('ovo')}},
            'Naive Bayes': {'func': BernoulliNB, 'hyper':{}},
            'MLP': {'func': MLPClassifier, 'hyper':{'alpha':[1e-5], 'hidden_layer_sizes':[[100, 100, 100]]}},
            'Logistic Regression': {'func': LogisticRegression, 'hyper':{'max_iter':[4000]}},
            'Random Forest': {'func': RandomForestClassifier, 'hyper':{'max_features':('sqrt')}},
            'Decision Tree': {'func': DecisionTreeRegressor, 'hyper':{'max_features':('sqrt')}},
            'Adaboost Decision Tree': {'func': AdaBoostRegressor, 'hyper':{'base_estimator':[DecisionTreeRegressor(max_depth=4)], 'n_estimators':[300]}},
            'Gaussian Naive Bayes': {'func': GaussianNB, 'hyper':{}},
        }

      