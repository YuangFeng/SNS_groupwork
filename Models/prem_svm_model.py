

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV



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


'''
# Find the best hyperparameters for the SVM using gridsearchCV
# create SVM classifier
svm = SVC()
grid = {'C': [0.2, 0.5, 1, 5, 10],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
# create grid search object
grid_search = GridSearchCV(svm, grid, scoring='accuracy',cv=5)

# fit grid search to the data
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
y_pred=model.predict(X_test)

# print best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
'''


#Create a svm Classifier
clf= svm.SVC(kernel='rbf',C = 0.2,probability=True) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict
y_pred = clf.predict(X_test)


'''
# Evaluation metrics
print("SVC Accuracy:",metrics.accuracy_score(y_test, y_pred))  # acc: 0.56
print("Macro F1 score: ",f1_score(y_test, y_pred, average='macro'))
print("AUC ROC: ", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo'))

print("Accuracy for SVC on CV data: ",accuracy_score(y_test,y_pred))
print("Accuracy for SVC on training data: ",accuracy_score(y_train,clf.predict(X_train)))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=clf.classes_)
disp.plot()
plt.show()
'''


















































