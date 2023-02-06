import pandas as pd
import numpy as np

df = pd.read_csv('PremData - complete data.csv')
shape = np.shape(df)

def findpast(season, H_names, A_names, find_names, x):
    loc = df[(df['Season'] == season) & (df['H'] == H_names) & (df['A'] == A_names)]
    if loc.size == 0:
        return None
    index = loc.index.tolist()[0]
    count = 0
    goals = []
    goal_difference = []
    yellow_card = []
    for i in range(index + 1, len(df)):
        if df.iloc[i]['A'] == find_names:
            goals.append(df.iloc[i]['A Goals'])
            goal_difference.append(df.iloc[i]['A Goals'] - df.iloc[i]['H Goals'])
            yellow_card.append(df.iloc[i]['A yellows'])
        elif df.iloc[i]['H'] == find_names:
            goals.append(df.iloc[i]['H Goals'])
            goal_difference.append(df.iloc[i]['H Goals'] - df.iloc[i]['A Goals'])
            yellow_card.append(df.iloc[i]['H yellows'])
        else:
            continue
        count += 1
        if count == x:
            return sum(goals) , sum(goals)/x, sum(goal_difference), sum(goal_difference)/x, sum(yellow_card), sum(yellow_card)/x
            break
    if count == 0:
        return None

result_H = {}
result_A = {}

for i in range(0, 199):
    Hteam = df.iloc[i,2]
    Ateam = df.iloc[i,3]
    result_H[i] = findpast(22, Hteam, Ateam, Hteam, 5)
    result_A[i] = findpast(22, Hteam, Ateam, Ateam, 5)

for i in range(199, len(df)):
    Hteam = df.iloc[i,2]
    Ateam = df.iloc[i,3]
    result_H[i] = findpast(21, Hteam, Ateam, Hteam, 5)
    result_A[i] = findpast(21, Hteam, Ateam, Ateam, 5)

print(result_H)
print(result_A)