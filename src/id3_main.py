#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by ID3 ##################

import utils
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict,KFold,cross_val_score, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()

    # start timer
    startTime = datetime.datetime.now()

    # training using ID3
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf.fit(X_train,y_train)

    # results = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=4)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))

    # make predictions for test data
    y_pred = clf.predict(X_test)

    # end timer
    endTime = datetime.datetime.now()
    timeElapsed = endTime - startTime
    print('Total elapsed time for ID3 = ' + str(round(timeElapsed.total_seconds() * 1000, 4)) + ' ms')

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)

    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    pd.Series(y_test).value_counts()
    1-np.mean(y_test)
    print('True:', y_test[0:25])
    print('Pred:', y_pred[0:25])

    # print confusion matrix
    utils.printConfusionMatrix(y_test, y_pred)

    # plot ROC and ROC curves
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    utils.printCurves(X_test, y_test, y_pred, y_pred_prob, os.path.join('..','output','id3_curve.pdf'))
    
