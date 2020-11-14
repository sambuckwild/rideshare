import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from clean import *
from sklearn import neighbors
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, \
    mean_squared_error


def knn(k, X_train, X_test, y_train, y_test ):
    classifier = KNeighborsClassifier(k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title('Confusion Matrix - KNN', fontsize=16)
    plt.xlabel('Predicted Label - Churn= 1', fontsize=14)
    plt.ylabel('True Label - Churn= 1', fontsize=14)
    plt.savefig('confusion_matrix_11')
    plt.show()


def mse_plot(d):
    error = []

    for i in range(1, d):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, d), error, color='red', linestyle='dashed', marker='o',
                markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value', fontsize=20)
    plt.xlabel('K Value', fontsize=17)
    plt.ylabel('Mean Error', fontsize=17)
    plt.savefig('mse_plot.png')
    plt.show()


def report():
    reg = KNeighborsClassifier(n_neighbors=11)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))


if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    df = to_date(df)
    df = create_churn_col(df, 'last_trip_date', '2014-06-01')
    df = bool_to_int(df, ['churn', 'luxury_car_user'])
    df = hot_encode(df)
    df = drop_nan_ratings(df)
    df = drop_cols(df)

    y = df.pop('churn').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    knn(11, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    knn(12, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    mse_plot(25)
    report()
