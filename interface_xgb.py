import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import MinMaxScaler

from common import findpast, findpast_homeonly, findpast_awayonly, meetingrecord, countstreak, streak


def predictfuture(season, H_name, A_name):
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
    xgb_model = joblib.load('./trained_xgb.sav')  # load the trained XGB model

    df1 = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')  # load preprocessed data
    df1 = df1.dropna()  # data set cleaning

    Hteam = H_name
    Ateam = A_name

    df2 = pd.DataFrame(columns=df1.columns) # A new dataframe to store new features for furture match prediction

    new_row = [season, None, H_name, A_name,
               None, None, None, None,
               None, None, None, None,
               None, None, None, None,
               None, None, None, findpast(season, Hteam, Ateam, Hteam, 5)[0],
               findpast(season, Hteam, Ateam, Hteam, 5)[1], findpast(season, Hteam, Ateam, Hteam, 5)[2], findpast(season, Hteam, Ateam, Hteam, 5)[3], findpast(season, Hteam, Ateam, Hteam, 5)[4],
               findpast(season, Hteam, Ateam, Hteam, 5)[5], findpast(season, Hteam, Ateam, Hteam, 5)[6], findpast(season, Hteam, Ateam, Ateam, 5)[0], findpast(season, Hteam, Ateam, Ateam, 5)[1],
               findpast(season, Hteam, Ateam, Ateam, 5)[2], findpast(season, Hteam, Ateam, Ateam, 5)[3], findpast(season, Hteam, Ateam, Ateam, 5)[4], findpast(season, Hteam, Ateam, Ateam, 5)[5],
               findpast(season, Hteam, Ateam, Ateam, 5)[6], findpast(season, Hteam, Ateam, Hteam, 20)[0], findpast(season, Hteam, Ateam, Hteam, 20)[1], findpast(season, Hteam, Ateam, Hteam, 20)[2],
               findpast(season, Hteam, Ateam, Hteam, 20)[3], findpast(season, Hteam, Ateam, Hteam, 20)[4], findpast(season, Hteam, Ateam, Hteam, 20)[5], findpast(season, Hteam, Ateam, Hteam, 20)[6],
               findpast(season, Hteam, Ateam, Ateam, 20)[0], findpast(season, Hteam, Ateam, Ateam, 20)[1], findpast(season, Hteam, Ateam, Ateam, 20)[2], findpast(season, Hteam, Ateam, Ateam, 20)[3],
               findpast(season, Hteam, Ateam, Ateam, 20)[4], findpast(season, Hteam, Ateam, Ateam, 20)[5], findpast(season, Hteam, Ateam, Ateam, 20)[6], meetingrecord(season, Hteam, Ateam, Ateam, 3)[0],
               meetingrecord(season, Hteam, Ateam, Ateam, 3)[1], meetingrecord(season, Hteam, Ateam, Hteam, 3)[0], meetingrecord(season, Hteam, Ateam, Hteam, 3)[1], streak(season, Hteam, Ateam, Hteam, 5)[0],
               streak(season, Hteam, Ateam, Hteam, 5)[1], streak(season, Hteam, Ateam, Hteam, 5)[2], streak(season, Hteam, Ateam, Ateam, 5)[0], streak(season, Hteam, Ateam, Ateam, 5)[1],
               streak(season, Hteam, Ateam, Ateam, 5)[2], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[0], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[1], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[2],
               findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[3], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[4], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[5], findpast_homeonly(season, Hteam, Ateam, Hteam, 5)[6],
               findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[0], findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[1], findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[2], findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[3],
               findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[4], findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[5],findpast_awayonly(season, Hteam, Ateam, Ateam, 5)[6]] #Create new features to predict future match

    df2.loc[0, :] = new_row

    df = pd.concat([df2, df1])#merge the created frame with the dataset
    all_features = df[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_points5', 'A_points5', \
                   'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_points20',
                   'A_points20', \
                   'H_goal_difference3', 'A_goal_difference3', \
                   'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5', 'A_away_points5',
                   'H_home_goals5', \
                   'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5', \
                   'H_W_streaks', 'H_D_streaks', 'H_L_streaks', 'A_W_streaks', 'A_D_streaks', 'A_L_streaks']].copy() #All features in dataset, used to scaling

    loc = df[(df['Season'] == season) & (df['H'] == H_name) & (df['A'] == A_name)]  ## locate the predict match, it should be the first row of df
    x = loc[['H_avg_goals5', 'A_avg_goals5', 'H_goal_difference5', 'A_goal_difference5', 'H_points5', 'A_points5', \
             'H_avg_goals20', 'A_avg_goals20', 'H_goal_difference20', 'A_goal_difference20', 'H_points20', 'A_points20', \
             'H_goal_difference3', 'A_goal_difference3', \
             'H_home_goal_difference5', 'A_away_goal_difference5', 'H_home_points5', 'A_away_points5', 'H_home_goals5', \
             'A_away_goals5', 'H_home_avg_goals5', 'A_away_avg_goals5', \
             'H_W_streaks', 'H_D_streaks', 'H_L_streaks', 'A_W_streaks', 'A_D_streaks', 'A_L_streaks']].copy()#The input features of XGBoost

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(all_features)
    x_scaled = scaler.transform(x)  # Scale the data (not 100% necassary)

    y_pred = xgb_model.predict(x_scaled)
    # print("y_pred is " + str(y_pred))

    if y_pred == 0:
        return "The winner is " + H_name  # Home team win
    elif y_pred == 1:
        return 'Match Draws'  # Draw
    elif y_pred == 2:
        return "The winner is " + A_name  # Away team win
    else:
        return "Can not predict due to insuffcient data"  # Can not predict due to insuffcient data


print(predictfuture(23, 'Man City', 'Liverpool'))# The game will be played on 1st April, Model think Man City will win
