#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import metrics


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