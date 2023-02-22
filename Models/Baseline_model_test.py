# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:22:42 2023

@author: solom
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm


# Load preprocessed data
df = pd.read_csv('1502b.csv')
shape = np.shape(df)
df = df.dropna()


# Convert 'Result' H/D/A to numeric 1/0/-1 for Machine Learning purposes

arr = []
results = df['Result']
arr_res = np.array(results)

res = 10000
for i, iter in enumerate(arr_res):
    if arr_res[i] == 'H':
        res = 0
        arr.append(res)
    elif arr_res[i] == 'D':
        res = 1
        arr.append(res)
    elif arr_res[i] == 'A':
        res = 2
        arr.append(res)

df['Result_numeric']=arr



# Machine learning modelling begins

label = df['Result_numeric']
features = df[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_avg_yellow_card5', 'A_avg_yellow_card5', 'H_points5','A_points5',\
               'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_avg_yellow_card20', 'A_avg_yellow_card20', 'H_points20','A_points20',\
                   'H_goal_difference3', 'A_goal_difference3', 'H_avg_goal_difference3', 'A_avg_goal_difference3',\
                       'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5','A_away_points5', 'H_home_goals5', 'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5']].copy()

X= features  # Features
y= label  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(7)) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
# Predict
y_pred=clf.predict(X_test)

# Model Accuracy
print("RF Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Try with linear SVC

#Create a svm Classifier
clf2 = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf2.fit(X_train, y_train)

#Predict
y_pred2 = clf2.predict(X_test)

# Model Accuracy:
print("SVC Accuracy:",metrics.accuracy_score(y_test, y_pred2))



# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm, clf.classes).plot()

# Try with basic XGBoost classifier


# import xgboost
# from xgboost import XGBClassifier
# # fit model no training data
# model = XGBClassifier()
# model.fit(X_train, y_train)


# # make predictions for test data
# y_pred3 = model.predict(X_test)
# predictions = [round(value) for value in y_pred3]

# from sklearn.metrics import accuracy_score


# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracyxg: %.2f%%" % (accuracy * 100.0))










