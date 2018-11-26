#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by XGBoost ##################

import utils
import datetime
import xgboost as xgb
from matplotlib.pyplot import show

if __name__ == '__main__':

    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()

    # start timer
    startTime = datetime.datetime.now()

    # fit model
    train_data = xgb.DMatrix(X_train, y_train, feature_names=feature_labels)
    train_param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
    train_iteration = 100
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
    xgb.plot_tree(model)
    show()

