#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 3/28/19

"""
eval_results.py
 * calculate costs and max_costs
 * evaluate results (called from main.py -> print results to console)
 * define score_and_costs for results.ipynb
"""

import numpy as np
from sklearn.metrics import f1_score

def costs(y_true, y_pred, cost_mat):
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2]) +\
    (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.round(sum(cost), 2)

def max_costs(y_true, cost_mat):
    cost_all_neg = sum(y_true * cost_mat[:, 1] + (1 - y_true) * cost_mat[:, 3])
    return cost_all_neg
    
def savings(X_train, y_train, X_test, y_test, cost_mat_train, cost_mat_test, model):
    cost_train = costs(y_train, model.predict(X_train), cost_mat_train)
    max_cost_train = max_costs(y_train, cost_mat_train)
    cost_test = costs(y_test, model.predict(X_test), cost_mat_test)
    max_cost_test = max_costs(y_test, cost_mat_test)
    # max_cost is if we would label everything as negative (i.e. if we don't use our model)
    # costs calculates how much costs our model makes (i.e. we still make costs but less since we catch some frauds)
    # this ratio then shows the savings of our model vs. not using our model at all
    train_savings = np.round((max_cost_train - cost_train) / max_cost_train, 3)
    test_savings = np.round((max_cost_test - cost_test) / max_cost_test, 3)
    return (train_savings, test_savings)

def evaluate(name, y_train, y_test, y_pred_train, y_pred_test, cost_mat_train, cost_mat_test):
    data = zip(y_train, y_test, y_pred_train, y_pred_test, cost_mat_train, cost_mat_test)
    pos_train, pos_test = [], []
    f1_train, f1_test = [], []
    cost_savings_train, cost_savings_test = [], []
    for y_train, y_test, y_pred_train, y_pred_test, cost_mat_train, cost_mat_test in data:
        pos_train.append(np.round(y_pred_train.sum() / y_pred_train.shape[0], 3))
        pos_test.append(np.round(y_pred_test.sum() / y_pred_test.shape[0], 3))
        f1_train.append(np.round(f1_score(y_train, y_pred_train), 3))
        f1_test.append(np.round(f1_score(y_test, y_pred_test), 3))
        max_cost_train = max_costs(y_train, cost_mat_train)
        cost_train = costs(y_train, y_pred_train, cost_mat_train)
        cost_savings_train.append(np.round((max_cost_train - cost_train) / max_cost_train, 3))
        max_cost_test = max_costs(y_test, cost_mat_test)
        cost_test = costs(y_test, y_pred_test, cost_mat_test)
        cost_savings_test.append(np.round((max_cost_test - cost_test) / max_cost_test, 3))
    print('--------------------')
    print(name)
    print('Positives train     ' + str(np.round(np.mean(pos_train), 3)) +
          ', individual: ' + str(pos_train))
    print('Positives test:     ' + str(np.round(np.mean(pos_test), 3)) + 
          ', individual: ' + str(pos_test))
    print('F1-score train:     ' + str(np.round(np.mean(f1_train), 3)) +
          ', individual: ' + str(f1_train))
    print('F1-score test:      ' + str(np.round(np.mean(f1_test), 3)) +
          ', individual: ' + str(f1_test))
    print('Cost savings train: ' + str(np.round(np.mean(cost_savings_train), 3)) + 
          ', individual: ' + str(cost_savings_train))
    print('Cost savings test:  ' + str(np.round(np.mean(cost_savings_test), 3)) + 
          ', individual: ' + str(cost_savings_test))
    
def scores_and_costs(y_true_l, y_pred_l, cost_mat):
    f1_scores, cost_savings = [], []
    for y_true, y_pred, cost_mat in zip(y_true_l, y_pred_l, cost_mat):
        f1_scores.append(np.round(f1_score(y_true, y_pred), 3))
        max_costs_ = max_costs(y_true, cost_mat)
        costs_ = costs(y_true, y_pred, cost_mat)
        cost_savings.append(np.round((max_costs_ - costs_) / max_costs_, 3))
    return f1_scores, cost_savings