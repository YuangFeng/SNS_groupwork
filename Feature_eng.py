import pandas as pd

df = pd.read_csv('PremData - complete data.csv')#read data

def findpast(season, H_name, A_name, find_name, x):#Function return new features of single team in past x matches
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

def meetingrecord(season, H_name, A_name, findname, x):#Function that return  meeting records of past x matches between two teams
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
            return sum(goal_difference), sum(goal_difference)/x
            break
    if count == 0:
        return None


result_H_past5 = {}
result_A_past5 = {}
result_H_past20 = {}
result_A_past20 = {}
Meeting_record_A_3 = {}
Meeting_record_H_3 ={}## Dictionaries to store features

season_cutoff = [199, 579, 959, 1219, 1449, 1599, 1979, 2359, 2739, 3119, 3499, 3879, 4259, 4639] # no. of rows which a new season start
season = 23

for i in range(0, len(df)):# Looping row by row to create new features
#    if i > 10:
#       break
    for j in season_cutoff:
        if i == j:
            season = season - 1
    Hteam = df.iloc[i,2]
    Ateam = df.iloc[i,3]
    result_H_past5[i] = findpast(season, Hteam, Ateam, Hteam, 5)
    result_A_past5[i] = findpast(season, Hteam, Ateam, Ateam, 5)
    result_H_past20[i] = findpast(season, Hteam, Ateam, Hteam, 20)
    result_A_past20[i] = findpast(season, Hteam, Ateam, Ateam, 20)
    Meeting_record_A_3[i] = meetingrecord(season, Hteam, Ateam, Ateam, 3)
    Meeting_record_H_3[i] = meetingrecord(season, Hteam, Ateam, Hteam, 3)
    
# Initializing lists for storing dictionary data
goals = []
avg_goals = []
goal_difference = []
avg_goal_difference = []
yellow_card = []
avg_yellow_card = []
points = []
lists = [goals, avg_goals, goal_difference, avg_goal_difference, yellow_card, avg_yellow_card, points]
past_goal_difference = []
avg_past_goal_difference = []
lists2 = [past_goal_difference, avg_past_goal_difference]


# Load past results from dictionary to lists
def load_pastresults(lists, result):
    for lst in lists:
        lst.clear()  # clear them before use

    for i in range(len(result)):  # Iterate each element in dictionary
        if result[i] is None:
            for lst in lists:
                lst.append(None)
        else:
            for j, lst in enumerate(lists):
                lst.append(result[i][j])


def load_meetingrecord(lists, result):
    for lst in lists:
        lst.clear()  # clear them before use

    for i in range(len(result)):  # Iterate each element in dictionary
        if result[i] is None:
            for lst in lists:
                lst.append(None)
        else:
            for j, lst in enumerate(lists):
                lst.append(result[i][j])

load_pastresults(lists, result_H_past5)
df = df.assign(H_goals5=goals, H_avg_goals5=avg_goals, H_goal_difference5=goal_difference,
               H_avg_goal_difference5=avg_goal_difference, H_yellow_card5=yellow_card,
               H_avg_yellow_card5=avg_yellow_card, H_points5=points)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_pastresults(lists, result_A_past5)
df = df.assign(A_goals5=goals, A_avg_goals5=avg_goals, A_goal_difference5=goal_difference,
               A_avg_goal_difference5=avg_goal_difference, A_yellow_card5=yellow_card,
               A_avg_yellow_card5=avg_yellow_card, A_points5=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_pastresults(lists, result_H_past20)
df = df.assign(H_goals20=goals, H_avg_goals20=avg_goals, H_goal_difference20=goal_difference,
               H_avg_goal_difference20=avg_goal_difference, H_yellow_card20=yellow_card,
               H_avg_yellow_card20=avg_yellow_card, H_points20=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_pastresults(lists, result_A_past20)
df = df.assign(A_goals20=goals, A_avg_goals20=avg_goals, A_goal_difference20=goal_difference,
               A_avg_goal_difference20=avg_goal_difference, A_yellow_card20=yellow_card,
               A_avg_yellow_card20=avg_yellow_card, A_points20=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_meetingrecord(lists2, Meeting_record_A_3)
df = df.assign(A_goal_difference3 = past_goal_difference, A_avg_goal_difference3 = avg_past_goal_difference)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_meetingrecord(lists2, Meeting_record_H_3)
df = df.assign(H_goal_difference3 = past_goal_difference, H_avg_goal_difference3 = avg_past_goal_difference)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)
