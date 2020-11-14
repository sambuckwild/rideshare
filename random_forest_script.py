from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def drop_str_dt(df):
    #Dropping Columns that are strings and date_time objects
    df = df.drop(['city', 'last_trip_date', 'signup_date'], axis=1, inplace=True)
    return df


def random_forest(df,n_estimators=n_estimators):

    #Creating target y (Churn: True or False)
    y = df.pop('churn').values

    #Creating X values 
    X = df.values

    #Performing a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Creating random forest and fitting it
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    #Get accuracy score for RF with n_estimator amount of trees
    accuracy = rf.score(X_test, y_test)

    #Getting feature importances
    importances = rf.feature_importances_

    #Casting column names to list
    cols = df.columns.tolist()

    #Creating dictionary with col name and importance value
    importance_dict = {}
    count = 0
    for i in cols:
        importance_dict[i] = importances[count]
        count += 1
    
    return accuracy, importance_dict

def num_tree_plot(df)
    num_trees = range(5, 50, 5)
    accuracies = []
    for n in num_trees:
        tot = 0
        for i in range(5):
            rf = RandomForestClassifier(n_estimators=n)
            rf.fit(X_train, y_train)
            tot += rf.score(X_test, y_test)
        accuracies.append(tot / 5)
    fig, ax = plt.subplots()
    ax.plot(num_trees, accuracies)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title('Accuracy vs Num Trees')


if __name__ == '__main__':
    num_tree_plot(df)