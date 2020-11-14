import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics, model_selection
from clean import *

def get_importances(model, n):
    importances = rf.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],    axis = 0)
    indices = np.argsort(importances)[::-1]
    features = list(df.columns[indices])
    return importances, std, indices, features 

def plot_importances(model, n):
    '''
    Plots the impurity-based feature importances of the forest

    Parameters:
        model: (var): the random forest model instantiated 
        n: (int) how many importances do we want to look at

    Returns:
        a bar graph of the feature importances using a random forest model
    '''
    importances = model.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for tree in model.estimators_],    axis = 0)
    indices = np.argsort(importances)[::-1]
    features = list(df.columns[indices])

    fig, ax = plt.subplots()

    ax.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    ax.set_xticks(range(n))
    ax.set_xticklabels(features, rotation = 90)
    ax.set_xlim([-1, n])
    ax.set_xlabel("importance")
    ax.set_title("Feature Importances")


if __name__ == "__main__":
    
    # Print the feature ranking
    rf = RandomForestClassifier()
    #more rf stuff here
    features = get_importances(rf, n)[-1]
    importances = get_importances(model, n)[0]
    indices = get_importances(model, n)[2]
    # print("\nLIST: Feature ranking:")
    # for f in range(n):
    #     print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

    num_trees = range(5, 200, 5)
    accuracies = []
    for n in num_trees:
    tot = 0
    for i in range(5):
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    fig, ax = plt.subplots()
    ax.plot(num_trees, accuracies)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title('Accuracy vs Num Trees')


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


    from roc import plot_roc
    plot_roc(X, y, RandomForestClassifier, 'Random_Forest', n_estimators=25, max_features=5)
    plot_roc(X, y, LogisticRegression, 'Logistic_Regression')
    plot_roc(X, y, DecisionTreeClassifier, 'Decision_Tree')