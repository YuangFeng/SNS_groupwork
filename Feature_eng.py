import pandas as pd
import numpy as np

df = pd.read_csv('PremData - complete data.csv')
shape = np.shape(df)

def findpast(season, H_name, A_name, find_name, x):
    loc = df[(df['Season'] == season) & (df['H'] == H_name) & (df['A'] == A_name)]
    if loc.size == 0:
        return None
    index = loc.index.tolist()[0]
    count = 0
    goals = []
    goal_difference = []
    yellow_card = []
    points = []
    for i in range(index + 1, len(df)):
        if df.iloc[i]['A'] == find_name:
            goals.append(df.iloc[i]['A Goals'])
            goal_difference.append(df.iloc[i]['A Goals'] - df.iloc[i]['H Goals'])
            yellow_card.append(df.iloc[i]['A yellows'])
            points.append(df.iloc[i]['A points'])
        elif df.iloc[i]['H'] == find_name:
            goals.append(df.iloc[i]['H Goals'])
            goal_difference.append(df.iloc[i]['H Goals'] - df.iloc[i]['A Goals'])
            yellow_card.append(df.iloc[i]['H yellows'])
            points.append(df.iloc[i]['H points'])
        else:
            continue
        count += 1
        if count == x:
            return sum(goals) , sum(goals)/x, sum(goal_difference), sum(goal_difference)/x, sum(yellow_card), sum(yellow_card)/x, sum(points)
            break
    if count == 0:
        return None

def meetingrecord(season, H_name, A_name, findname, x):
    loc = df[(df['Season'] == season) & (df['H'] == H_name) & (df['A'] == A_name)]
    if loc.size == 0:
        return None
    index = loc.index.tolist()[0]
    count = 0
    goal_difference = []
    for i in range(index + 1, len(df)):
        if ((df.iloc[i]['A'] == H_name and df.iloc[i]['H'] == A_name) or (df.iloc[i]['A'] == A_name and df.iloc[i]['H'] == H_name)) and df.iloc[i]['H'] == findname:
            goal_difference.append(df.iloc[i]['H Goals'] - df.iloc[i]['A Goals'])
        elif ((df.iloc[i]['A'] == H_name and df.iloc[i]['H'] == A_name) or (df.iloc[i]['A'] == A_name and df.iloc[i]['H'] == H_name)) and df.iloc[i]['A'] == findname:
            goal_difference.append(df.iloc[i]['A Goals'] - df.iloc[i]['H Goals'])
        else:
            continue
        count += 1
        if count == x:
            return goal_difference
            break
    if count == 0:
        return None


result_H_past5 = {}
result_A_past5 = {}
result_H_past20 = {}
result_A_past20 = {}
Meeting_record_A_2 = {}
Meeting_record_H_2 ={}

season_cutoff = [199, 579, 959, 1219, 1449, 1599, 1979, 2359, 2739, 3119, 3499, 3879, 4259, 4639]
season = 23

for i in range(0, len(df)):
    for j in season_cutoff:
        if i == j:
            season = season - 1
    Hteam = df.iloc[i,2]
    Ateam = df.iloc[i,3]
    result_H_past5[i] = findpast(season, Hteam, Ateam, Hteam, 5)
    result_A_past5[i] = findpast(season, Hteam, Ateam, Ateam, 5)
    result_H_past20[i] = findpast(season, Hteam, Ateam, Hteam, 20)
    result_A_past20[i] = findpast(season, Hteam, Ateam, Ateam, 20)
    Meeting_record_A_2[i] = meetingrecord(season, Hteam, Ateam, Ateam, 3)
    Meeting_record_H_2[i] = meetingrecord(season, Hteam, Ateam, Hteam, 3)

print(result_H_past5)
print(result_A_past20)
print(Meeting_record_A_2)























































































