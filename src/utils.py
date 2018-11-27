#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


def loadDataSet():
    # load the dataset
    dataset = pd.read_csv(os.path.join('..','data','data.csv'))
    dataset.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
    dataset.diagnosis = [1 if each == "M" else 0 for each in dataset.diagnosis]
    y = dataset.diagnosis.values
    dataset.drop(['diagnosis'], axis=1, inplace=True)
    x = dataset.values

    # split data into training set and testing set 80% vs 20%
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test, dataset.columns

def printAccuarcy(y_test, y_pred):
    predictions = [round(value) for value in y_pred]
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def printConfusionMatrix(y_test, y_pred):
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    print(conf_mat)
    TP = conf_mat[1,1]
    TN = conf_mat[0,0]
    FP = conf_mat[0,1]
    FN = conf_mat[1,0]

    # Sensitivity: When the actual value is posiitive, how often is the prediction correct
    # How "sensitive" is the classifier to detecting positive instances
    print("Sensitivity: When the actual value is posiitive, how often is the prediction correct")
    print("Sensitivity: %0.2f%%" % (TP / float(TP + FN) * 100.0))

    print("Specificity: When the actual value is negative, how often is the prediction correct")
    print("Specificity: %0.2f%%" % (TN / float(TN+FP) * 100.0))

    print("False Positive Rate: When the actual value is negative, how often is the prediction incorrect?")
    print("False +ve Rate: %0.2f%%" % (FP / float(TN+FP) * 100.0))

    print("Precision: When a positive value is predicted, how often is the prediction correct?")
    print("Precision: %0.2f%%" % (TP / float(TP+FP) * 100.0))

    print("Accuracy: How often is the prediction correct overall")
    print("Accuracy %0.2f%%" % ((TP + TN) / float(TP + TN + FP + FN) * 100.0))

def printCurves(X_test, y_test, y_pred, y_pred_prob, filePath):
    # ROC Curves and Area Under the Curve (AUC)
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)

    plt.plot(fpr,tpr)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title("ROC Curve for Breast Cancer Detection")
    plt.xlabel("False Positive Rate (1-Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.savefig(filePath, bbox_inches='tight')

    print("AUC is the percentage of the ROC plot that underneath the curve:")
    print("AUC Score: %0.2f%%" % (metrics.roc_auc_score(y_test,y_pred) * 100.0))