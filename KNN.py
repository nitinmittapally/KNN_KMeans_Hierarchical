# -*- coding: utf-8 -*-
"""
Created on Sun May 23 20:53:39 2021

@author: nitin
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import seaborn as sns

iris = datasets.load_iris()

data = pd.DataFrame(np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["species"])
print(iris["DESCR"])

data["species"] = data["species"].astype("category")
data.describe()
data.head()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"],
}

from sklearn.model_selection import GridSearchCV

scoring = ["precision_micro", "f1_micro", "accuracy", "recall_micro"]
gridSearchCV = GridSearchCV(classifier, parameters, cv=10, scoring=scoring,
                            verbose=1,return_train_score=True, refit="f1_micro")
gridSearchCV.fit(X_train, y_train)
gridSearchCV.best_estimator_
results = gridSearchCV.cv_results_
test_score = pd.DataFrame(np.c_[range(1,50), 
                                results["mean_test_f1_micro"][list(range(0,98,2))],
                                results["mean_test_f1_micro"][list(range(1,98,2))],
                                results["mean_train_f1_micro"][list(range(0,98,2))], 
                                results["mean_train_f1_micro"][list(range(1,98,2))]], 
                          columns=["k", "test_score_uniform","test_score_distance", "train_score_uniform", 
                                   "train_score_distance"])

fig, ax = plt.subplots(figsize=(20,16))
sns.lineplot(data = test_score, x = "k", y ="test_score_uniform" , ax = ax)
sns.lineplot(data = test_score, x = "k", y ="train_score_uniform" , ax = ax)

#fig, ax = plt.subplots(figsize=(20,16))
sns.lineplot(data = test_score, x = "k", y ="test_score_distance" , ax = ax)
sns.lineplot(data = test_score, x = "k", y ="train_score_distance" , ax = ax)

df = test_score.iloc[:,[0,1,3]].melt('k')
fig, ax = plt.subplots(figsize=(16,10))
sns.lineplot(data = df, x = "k", y ="value", style="variable")

df = test_score.iloc[:,[0,2,4]].melt('k')
fig, ax = plt.subplots(figsize=(16,10))
sns.lineplot(data = df, x = "k", y ="value", style="variable", hue="variable")

fig, ax = plt.subplots(1,2, figsize=(18,8))
df = test_score.iloc[:,[0,1,3]].melt('k')
sns.lineplot(data = df, x = "k", y ="value", style="variable", ax = ax[0], hue="variable")

df = test_score.iloc[:,[0,2,4]].melt('k')
sns.lineplot(data = df, x = "k", y ="value", style="variable", hue="variable", ax= ax[1])

from sklearn.metrics import classification_report, confusion_matrix

y_pred = gridSearchCV.predict(X_test)
y_test == y_pred
report = classification_report(y_test, y_pred, 
                            target_names=["Iris-Setosa","Iris-Versicolour","Iris-Virginica"]
                            )
print(confusion_matrix(y_test, y_pred))
print(report)
