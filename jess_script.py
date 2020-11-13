import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
from clean import *


def make_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 73)

rf = RandomForestClassifier()

importances_lst = np.argsort(rf.feature_importances_)
importances_lst = list(reversed(list(importances_lst)))
importances_lst

#get the standard deiation for the feature importances across all trees

n = 10 # get the top 10 features

importances = rf.feature_importances_[:n]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],    axis = 0)
indices = np.argsort(importances)[::-1]
features = list(df.columns[indices])

# Plot the impurity-based feature importances of the forest

fig, ax = plt.subplots()

ax.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
ax.set_xticks(range(10))
ax.set_xticklabels(features, rotation = 90)
ax.set_xlim([-1, 10])
ax.set_xlabel("importance")
ax.set_title("Feature Importances")

if __name__ == "__main__":
    pass