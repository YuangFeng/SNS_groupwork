import pandas as pd

def findpast(season, H_names, A_names, find_names, x):
    df = pd.read_csv('PremData - complete data.csv')
    loc = df[(df['Season'] == season) & (df['H'] == H_names) & (df['A'] == A_names)]
    if loc.size == 0:
        print('Not Found!')
        return
    index = loc.index.tolist()[0]
    count = 0
    for i in range(index + 1, len(df)):
        if df.iloc[i]['A'] == find_names:
            goals = df.iloc[i]['A Goals']
            goal_difference = df.iloc[i]['A Goals'] - df.iloc[i]['H Goals']
        elif df.iloc[i]['H'] == find_names:
            goals = df.iloc[i]['H Goals']
            goal_difference = df.iloc[i]['H Goals'] - df.iloc[i]['A Goals']
        else:
            continue
        print('{}, {}\tgoals:{}, goal difference:{}'.format(df.iloc[i]['Season'], df.iloc[i]['Data/Matchweek'], goals,
                                                            goal_difference))
        count += 1
        if count == x:
            break
    if count == 0:
        print('Not Found!')


findpast(21, 'Liverpool', 'Chelsea', 'Chelsea', 5)