#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### Breast cancer classification by ID3 ##################

import utils
import datetime
import id3
from graphviz import Source

if __name__ == '__main__':
    
    # load dataset
    X_train, X_test, y_train, y_test, feature_labels = utils.loadDataSet()

    # start timer
    startTime = datetime.datetime.now()
    
    # init Id3Estimator
    model = id3.Id3Estimator()

    # fit model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # end timer
    endTime = datetime.datetime.now()
    timeElapsed = endTime - startTime
    print('Total elapsed time for ID3 = ' + str(round(timeElapsed.total_seconds() * 1000, 4)) + ' ms')

    # evaluate predictions
    utils.printAccuarcy(y_test, y_pred)
    
    # plot tree
    id3.export_graphviz(model.tree_, '../output/id3_tree.dot', feature_labels)
    s = Source.from_file('../output/id3_tree.dot')
    s.view()
