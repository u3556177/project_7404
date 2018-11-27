#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by XGBoost ##################

import utils
import os
import datetime
import subprocess
import codecs
import xgboost as xgb
from matplotlib.pyplot import show
from graphviz import Digraph
from sys import platform

if __name__ == '__main__':

    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()

    # start timer
    startTime = datetime.datetime.now()

    # fit model
    train_data = xgb.DMatrix(X_train, y_train, feature_names=feature_labels)
    train_param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
    train_iteration = 50
    model = xgb.train(train_param, train_data, train_iteration)

    # make predictions for test data
    test_data = xgb.DMatrix(X_test, y_test, feature_names=feature_labels)
    y_pred = model.predict(test_data)

    # end timer
    endTime = datetime.datetime.now()
    timeElapsed = endTime - startTime
    print('Total elapsed time for XGBoost ' + str(train_iteration) + ' training iterations = ' + str(round(timeElapsed.total_seconds() * 1000, 4)) + ' ms')

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)

    # plot tree
    #xgb.plot_tree(model)
    #show()
    g = xgb.to_graphviz(model)
    f = codecs.open(os.path.join('..','output','xgb_tree.pdf'), mode='wb')
    f.write(g.pipe('pdf'))
    f.close()
    if platform == "linux" or platform == "linux2":
        subprocess.Popen('xdg-open ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)
    elif platform == "darwin":
        subprocess.Popen('open ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)
    elif platform == "win32":
        subprocess.Popen('start ' + os.path.join('..','output','xgb_tree.pdf'), shell=True)

