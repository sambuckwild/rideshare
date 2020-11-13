'''logisitic regression model'''
from clean import *
from initial_plots import image_of_plot
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, mean_squared_error, plot_roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=33)
    return X_train, X_test, y_train, y_test

def cross_val_kfold(n_splits, X, y):
    kf = KFold(n_splits=n_splits, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    accuracy = []
    precision = []
    recall = []
    mse = []

    for train_idx, test_idx in kf.split(X_train):
        model = LogisticRegression(max_iter=500)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model.predict(X_train.iloc[test_idx])
        y_true = y_train.iloc[test_idx]
        accuracy.append(accuracy_score(y_true, y_pred))
        precision.append(precision_score(y_true, y_pred))
        recall.append(recall_score(y_true, y_pred))
        mse.append(mean_squared_error(y_true, y_pred))

    return y_pred, y_true, accuracy, precision, recall, mse, model

def conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm

def plot_cm(cm, color, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=color)
    disp.im_
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label - Churn=1', fontsize=14)
    plt.ylabel('True Label - Churn=1', fontsize=14)

if __name__ == '__main__':
    file = (input("Enter path of filename: "))
    df = clean_this_df(file)
    y = df.pop('churn')
    X = df
    n=10
    y_pred, y_true, accuracy, precision, recall, mse, model = cross_val_kfold(
        X=X, y=y, n_splits=n)
    print('Average accuracy: {}'.format(np.mean(accuracy)))
    print('Average precision: {}'.format(np.mean(precision)))
    print('Average sensitivity: {}'.format(np.mean(recall)))
    print('Ave MSE: {}'.format(np.mean(mse)))
    cm = conf_matrix(y_true, y_pred)
    plot_cm(cm, 'Blues', 'Confusion Matrix - Logistic Regression')
    image_of_plot('images/log_cm.svg')
    for name, coef in zip(df.columns[1:], model.coef_[0]):
        print("{0}: {1:0.4f}".format(name, coef))