#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by ID3 ##################

import utils
import os
import datetime
import numpy as np
import pandas as pd
import graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

if __name__ == '__main__':

    # load dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

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

    # plot ROC and ROC curves
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    utils.printCurves(X_test, y_test, y_pred, y_pred_prob, os.path.join('..','output','id3_curve.pdf'))

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)

    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    pd.Series(y_test).value_counts()
    1-np.mean(y_test)
    print('True:', y_test[0:25])
    print('Pred:', y_pred[0:25])

    # print confusion matrix
    utils.printConfusionMatrix(y_test, y_pred)

    
    # plot tree
    dot_data = export_graphviz(clf, out_file=None, feature_names=dataset.feature_names, filled=True, 
    class_names=dataset.target_names ,rounded=True,  special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render(filename='id3_tree', directory=os.path.join('..','output'))

