

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import classification_report


# Load preprocessed data
df = pd.read_csv('1502b.csv')
shape = np.shape(df)
df = df.dropna()

# Convert 'Result' H/D/A to numeric 0/1/2 for Machine Learning purposes
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


### what if I just want to test if the home team win/dont win

# res = 10000
# for i, iter in enumerate(arr_res):
#     if arr_res[i] == 'H':
#         res = 0
#         arr.append(res)
#     elif arr_res[i] == 'D':
#         res = 1
#         arr.append(res)
#     elif arr_res[i] == 'A':
#         res = 1
#         arr.append(res)
####

df['Result_numeric']=arr

# Machine learning modelling begins   
    
label = df['Result_numeric']

features = df[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_points5','A_points5',\
                  'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_points20','A_points20',\
                      'H_goal_difference3', 'A_goal_difference3',\
                          'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5','A_away_points5', 'H_home_goals5',\
                              'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5',\
                                  'H_W_streaks','H_D_streaks','H_L_streaks','A_W_streaks','A_D_streaks','A_L_streaks']].copy()
 
X= features  # Features
y= label  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(7)) # 70% training and 30% test


# Scale the data (not 100% necassary)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters to search over
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200],
    #'min_child_weight': [1, 5, 10],
    #'gamma': [0.5, 1, 1.5, 2, 5],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.6, 0.8, 1.0],
}

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Train the XGBoost classifier with the best hyperparameters
best_xgb_model = xgb.XGBClassifier(**grid_search.best_params_)
best_xgb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = best_xgb_model.predict(X_test_scaled)

plot_importance(best_xgb_model)
pyplot.show()

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))







