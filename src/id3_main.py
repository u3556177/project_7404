#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by ID3 ##################

import utils
import id3
from graphviz import Source

if __name__ == '__main__':
    
    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()
    
    # init Id3Estimator
    model = id3.Id3Estimator()

    # fit model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)
    
    # plot tree
    id3.export_graphviz(model.tree_, '../output/id3_tree.dot', feature_labels)
    s = Source.from_file('../output/id3_tree.dot')
    s.view()
