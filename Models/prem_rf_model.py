

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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


# Find the best hyperparameters for the random forest using gridsearchCV

# rfc=RandomForestClassifier(random_state=42)

# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train, y_train)

# print(CV_rfc.best_params_)

########
########

# retrain forest model with tuned hyperparameters

rfc1=RandomForestClassifier(random_state=42, 
                            max_features='auto', 
                            n_estimators= 200, 
                            max_depth=7, 
                            criterion='gini')

rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_test)

print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
print(classification_report(y_test, pred))





















































