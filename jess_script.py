import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics, model_selection
from clean import *

def clean_this_df(file):
    '''
    cleans the dataframe according to clean.py
    '''
    df = pd.read_csv(file)
    df = to_date(df)
    df = create_churn_col(df, 'last_trip_date', '2014-06-01')
    df = bool_to_int(df, ['churn', 'luxury_car_user'])
    df = hot_encode(df)
    df = drop_nan_ratings(df)
    df.drop(['signup_date'])
    df.drop(['city', 'last_trip_date', 'signup_date'], axis=1, inplace=True)
    return df


# #get the importances and the standard deviation for the feature importances across all trees
# def get_importances(model, top_n)
#     n = top_n 
#     importances = rf.feature_importances_[:n]
#     std = np.std([tree.feature_importances_ for tree in rf.estimators_],    axis = 0)
#     indices = np.argsort(importances)[::-1]
#     features = list(df.columns[indices])
#     return importances, std, indices, features 


def plot_importances(model, top_n):
    '''
    Plots the impurity-based feature importances of the forest

    Parameters:
        model: (var): the random forest model instantiated 
        top_n: (int) how many importances do we want to look at

    Returns:
        a bar graph of the feature importances using a random forest model
    '''
    n = top_n 
    importances = rf.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],    axis = 0)
    indices = np.argsort(importances)[::-1]
    features = list(df.columns[indices])

    fig, ax = plt.subplots()

    ax.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    ax.set_xticks(range(10))
    ax.set_xticklabels(features, rotation = 90)
    ax.set_xlim([-1, 10])
    ax.set_xlabel("importance")
    ax.set_title("Feature Importances")


if __name__ == "__main__":
    pass
    df = clean_this_df('data/churn_train.csv')
    X = df.drop("churn", axis = 1)
    y = df['churn']

    #Adjust the kwargs/parameters accordingly to see if we can get a better model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #adjust this parameters 
    num_estimators = 10
    rf = RandomForestClassifier(n_estimators = num_estimators)
    rf.fit(X_train, y_train)
    predictions = rd.predict(X_test)
    score = rf.score(X_test, y_test)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_true = y_test, y_pred = predictions)
    print (f'At {num_etimators}: \n Confusion Matrix: {conf_matrix}, \n Accuracy = {score}, \n Precision = {precision}, \n Recall = {recall}')
