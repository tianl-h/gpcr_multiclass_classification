# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2020-12-15 15:29:10
# @Last Modified by:   tianl
# @Last Modified time: 2021-03-03 11:19:35


from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc, accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, precision_score, recall_score, balanced_accuracy_score
import numpy as np

def create_labels(stats):
    labels_by_names = {"Decoys": 0}
    m = 0
    for k in stats:
        if k == 'Decoys':
            continue
        m += 1
        labels_by_names[k] = m

    labels = []
    for comp, v in stats.items():
        labels += [labels_by_names[comp] for _ in range(v[0], v[1])]

    labels = np.array(labels)
    names_by_labels = {str(v): k for k, v in labels_by_names.items()}
    return labels, names_by_labels

def get_model(method, func, param=None):
    model = func() if param == None else func(**param)
    return model

def roc_auc_score_multiclass(y_test, y_test_pred_prime, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(y_test)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_y_test = [0 if x in other_class else 1 for x in y_test]
    new_y_test_pred_prime = [0 if x in other_class else 1 for x in y_test_pred_prime]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_y_test, new_y_test_pred_prime)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

def CalculateMetrics(y_test, y_test_pred, y_test_pred_prime, method):
    # if method == 'SVM':
    #     roc_auc = roc_auc_score(y_test, y_test_pred, multi_class="ovo",
    #                                   average="macro")
        
    if method == 'Decision Tree' or 'Random Forest' or 'SVM':
        roc_auc = roc_auc_score_multiclass(y_test, y_test_pred_prime)
        sum = 0
        for m, v in roc_auc.items():
            sum = sum + v
        roc_auc = sum / 11
    
    else:
        roc_auc = roc_auc_score(y_test, y_test_pred, multi_class="ovr",
                                      average="macro")

    ACC = accuracy_score(y_test, y_test_pred_prime.astype(int))
    Bal_ACC = balanced_accuracy_score(y_test, y_test_pred_prime.astype(int))
    F1_score = f1_score(y_test, y_test_pred_prime.astype(int), average = "macro")
    Cohan_Kappa = cohen_kappa_score(y_test, y_test_pred_prime.astype(int))
    MCC = matthews_corrcoef(y_test, y_test_pred_prime.astype(int))
    precision = precision_score(y_test, y_test_pred_prime.astype(int), average = "macro")
    recall = recall_score(y_test, y_test_pred_prime.astype(int), average = "macro")
    
    print("%.3f" % roc_auc ,"%.3f" % ACC, "%.3f" % Bal_ACC, "%.3f" % F1_score,"%.3f" % Cohan_Kappa, "%.3f" % MCC, "%.3f" % precision,"%.3f" % recall)
    return(roc_auc, ACC, Bal_ACC, F1_score, Cohan_Kappa, MCC, precision, recall)

