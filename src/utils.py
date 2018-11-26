#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def loadDataSet():
    # load the dataset 
    dataset = loadtxt('../data/data.csv', delimiter=",")
    # split data into X and y
    X = dataset[:,0:30]
    Y = dataset[:,30]
    # split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    labels = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
    'concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
    'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
    'concave_points_worst','symmetry_worst','fractal_dimension_worst']
    
    return X_train, X_test, y_train, y_test, labels

def printAccuarcy(y_test, y_pred):
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))