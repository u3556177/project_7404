#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by XGBoost ##################

import utils
import xgboost as xgb
from matplotlib.pyplot import show

if __name__ == '__main__':

    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()

    # fit model
    train_data = xgb.DMatrix(X_train, y_train, feature_names=feature_labels)
    train_param = {'max_depth':5, 'eta':0.1, 'silent':0, 'objective':'binary:logistic'}
    model = xgb.train(train_param, train_data)

    # make predictions for test data
    test_data = xgb.DMatrix(X_test, y_test, feature_names=feature_labels)
    y_pred = model.predict(test_data)

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)

    # plot tree
    xgb.plot_tree(model)
    show()

