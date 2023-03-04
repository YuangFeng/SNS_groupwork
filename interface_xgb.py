import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import MinMaxScaler

def predict(season, H_name, A_name):
    """
    This proposed function predict the result of matches, should be availavle to client via server
    Inputs:
        season: the season that a match will be play
        H_team: the home team name
        A_team: the away team name
    Three possible Outputs
        Home team name: indicates home team will win
        Draw: indicates the matche will draw
        Away team name: indicates away team will win
    """

    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')  # load preprocessed data
    df = df.dropna()#data set cleaning
    xgb_model = joblib.load('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Models/trained_xgb.sav')  # load the trained XGB model

    all_features = df[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_points5', 'A_points5', \
                   'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_points20',
                   'A_points20', \
                   'H_goal_difference3', 'A_goal_difference3', \
                   'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5', 'A_away_points5',
                   'H_home_goals5', \
                   'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5', \
                   'H_W_streaks', 'H_D_streaks', 'H_L_streaks', 'A_W_streaks', 'A_D_streaks', 'A_L_streaks']].copy()
    loc = df[(df['Season'] == season) & (df['H'] == H_name) & (df['A'] == A_name)]## locate the predict match
    x = loc[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_points5','A_points5',\
                  'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_points20','A_points20',\
                      'H_goal_difference3', 'A_goal_difference3',\
                          'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5','A_away_points5', 'H_home_goals5',\
                              'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5',\
                                  'H_W_streaks','H_D_streaks','H_L_streaks','A_W_streaks','A_D_streaks','A_L_streaks']].copy()
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(all_features)
    x_scaled = scaler.transform(x)# Scale the data (not 100% necassary)

    y_pred = xgb_model.predict(x_scaled)

    if y_pred == 0:
        return H_name #Home team win
    elif y_pred == 1:
        return 'Draw' #Draw
    elif y_pred == 2:
        return A_name #Away team win
    else:
        return None   #Can not predict due to insuffcient data

print(predict(23,'West Ham','Everton'))