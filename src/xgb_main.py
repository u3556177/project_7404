#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by XGBoost ##################

import utils
import os
import datetime
import codecs
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sys import platform

if __name__ == '__main__':

    # load dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
    feature_names = [w.replace(' ', '_') for w in dataset.feature_names]

    # start timer
    startTime = datetime.datetime.now()

    # fit model
    train_data = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    train_param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
    train_iteration = 10
    model = xgb.train(train_param, train_data, train_iteration)

    # make predictions for test data
    test_data = xgb.DMatrix(X_test, y_test, feature_names=feature_names)
    y_pred = model.predict(test_data)

    # end timer
    endTime = datetime.datetime.now()
    timeElapsed = endTime - startTime
    print('Total elapsed time for XGBoost ' + str(train_iteration) + ' training iterations = ' + str(round(timeElapsed.total_seconds() * 1000, 4)) + ' ms')

    # evaluate predictions
    y_pred = np.where(y_pred > 0.5, 1, 0)
    utils.printAccuarcy(y_test, y_pred)

    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    pd.Series(y_test).value_counts()
    1-np.mean(y_test)
    print('True:', y_test[0:25])
    print('Pred:', y_pred[0:25])

    # print confusion matrix
    utils.printConfusionMatrix(y_test, y_pred)

    # plot ROC and ROC curves
    classone_probs = y_pred
    classzero_probs = 1.0 - classone_probs
    y_pred_prob = np.vstack((classzero_probs, classone_probs)).transpose()[:,1]
    utils.printCurves(X_test, y_test, y_pred, y_pred_prob, os.path.join('..','output','xgb_curve.pdf'))

    # plot tree
    g = xgb.to_graphviz(model)
    f = codecs.open(os.path.join('..','output','xgb_tree.pdf'), mode='wb')
    f.write(g.pipe('pdf'))
    f.close()
    # if platform == "linux" or platform == "linux2":
    #     subprocess.Popen('xdg-open ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)
    # elif platform == "darwin":
    #     subprocess.Popen('open ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)
    # elif platform == "win32":
    #     subprocess.Popen('start ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)

