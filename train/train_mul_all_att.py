# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2020-12-15 13:50:32
# @Last Modified by:   tianl
# @Last Modified time: 2021-03-04 02:32:30

from utils.utils import create_labels
from utils.utils import get_model
from utils.utils import CalculateMetrics
from utils.models_1 import METHODS_WITH_HYPER
from utils.feature_reduction import feature_reduction

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly


import json
import os
import time
import pickle
import seaborn as sb

plotly.io.orca.config.executable = 'C:/Users/tianl/AppData/Local/Programs/orca/orca.exe'

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

    # create labels
    labels, names_by_labels = create_labels(stats)
    print(names_by_labels)

    # labels_ = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    metrics = {}

    X_train_prime, X_test, y_train_prime, y_test = train_test_split(data_all_features, labels,
                                                test_size=test_size, random_state=random_state)
        
    for method, func_and_hyper in METHODS_WITH_HYPER.items():
        start = time.time()
        func = func_and_hyper['func']
        hyper = func_and_hyper['hyper']

        model = get_model(method, func, param = hyper)
        
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)

        for train, test in cv.split(X_train_prime, y_train_prime):
            clf = model
            clf.fit(X_train_prime[train], y_train_prime[train])
            if method == 'Decision Tree':
                y_val_pred = clf.predict(X_train_prime[test])
                y_val_pred_prime = clf.predict(X_train_prime[test])

            else:
                y_val_pred = clf.predict_proba(X_train_prime[test])
                y_val_pred_prime = clf.predict(X_train_prime[test])


            roc_auc, ACC, Bal_ACC, F1_score, Cohan_Kappa, MCC, precision, recall = CalculateMetrics(y_train_prime[test], y_val_pred, y_val_pred_prime, method)
            
            if not method in metrics:
                metrics[method] = []

            metrics[method] = ["%.3f" % roc_auc,"%.3f" % ACC, "%.3f" % Bal_ACC, "%.3f" % F1_score, "%.3f" % Cohan_Kappa, "%.3f" % MCC, "%.3f" % precision,"%.3f" % recall]
            print(att + '_' + method + " test set:", roc_auc)
            print('%6s best hyper parameter: %30s, Accuracy: %.3f' % (method.upper(), str(best_param), ACC))
            stop = time.time()
            print("Training time: %.3f mins" % ((stop - start) / 60))


        ### Plot Radar Chart
        file = open(os.path.join('results_all_att/Validation_Radar/', "metrics_train.txt"), "w")

        

        for method, v in metrics.items():
            file.write(method + ':' + str(v))
            file.write('\t')
            file.write('\n')  
        file.close()

        layout = go.Layout(
    #     autosize=True,
        autosize=False,
        width=1000,
        height=800,
        margin=go.Margin(
        l=250,
        r=250)   
        
        )
        print(type(metrics['SVM'][0]))
        radar_chart = go.Figure(layout=layout)

        for method in metrics.keys():
            roc_auc = float(metrics[method][0])
            ACC = float(metrics[method][1])
            Bal_ACC = float(metrics[method][2])
            F1_score = float(metrics[method][3])
            Cohan_Kappa = float(metrics[method][4])
            MCC = float(metrics[method][5])
            precision = float(metrics[method][6])
            recall = float(metrics[method][7])

            radar_chart.add_trace(
                    go.Scatterpolar(
                        name = method,
                        r=[roc_auc, ACC, Bal_ACC, F1_score, Cohan_Kappa, MCC, precision, recall, roc_auc],
                        theta=['AUC', 'ACC', 'Bal_ACC', 'f1-Score', 'CK', 'MCC', 'Precision', 'Recall', 'AUC'],
                        showlegend=True
                        )
            )

            radar_chart.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=16)
                ),
                angularaxis = dict(
                    tickfont = dict(size = 20)
                ),
            )

            title= att + " on test set", font=dict(
            size=17)
            )

            radar_chart.write_image("results_all_att/Validation_Radar/" + 'All att'  + ".png")

if __name__ == '__main__':
    main()
