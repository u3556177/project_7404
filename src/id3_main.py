#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by ID3 ##################

import utils
from id3 import Id3Estimator
from id3 import export_graphviz
from sklearn.metrics import accuracy_score
from graphviz import Source

if __name__ == '__main__':
    
    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()
    
    # init Id3Estimator
    model = Id3Estimator()

    # fit model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate predictions
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # plot tree
    export_graphviz(model.tree_, 'output/id3_tree.dot', feature_labels)
    s = Source.from_file('output/id3_tree.dot')
    s.view()
