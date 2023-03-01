import pandas as pd

df = pd.read_csv('allSeasons - PremData - complete data.csv')#read data

def findpast(season, H_name, A_name, find_name, x):
    """
    This function will return number of new features to assess the performance(either at home or at away) of a specific team in the past.
    Input:
        season: the season of match
        H_name: the name of home team of match
        A_name: the name of away team of match
        find_name: the name of team that want to search(can only be H_name or A_name)
        x: number of past matches of find_name team
    Output:
        1.sum of goals of 'find_name' team in past x matches
        2.average goals of 'find_name' team in past x matches
        3.sum of goal difference of 'find_name' team in past x matches
        4.average goal difference of 'find_name' team in past x matches
        5.sum of yellow cards of 'find_name team' got in past x matches
        6.average of yellow cards of 'find_name team' got in past x matches
        7.sum of points that 'find_name team' obtained in past x matches
    """
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

def findpast_homeonly(season, H_name, A_name, find_name, x):
    """
    This function will return number of new features to assess the performance(at home only!) of a specific team in the past.
    Input:
        season: the season of match
        H_name: the name of home team of match
        A_name: the name of away team of match
        find_name: the name of team that want to search(can only be H_name or A_name)
        x: number of past matches of 'find_name' team
    Output:
        1.sum of goals of 'find_name' team in past x matches
        2.average goals of 'find_name' team in past x matches
        3.sum of goal difference of 'find_name' team in past x matches
        4.average goal difference of 'find_name' team in past x matches
        5.sum of yellow cards of 'find_name team' got in past x matches
        6.average of yellow cards of 'find_name team' got in past x matches
        7.sum of points that 'find_name team' obtained in past x matches
    """
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
        if df.iloc[i]['H'] == find_name:
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

def findpast_awayonly(season, H_name, A_name, find_name, x):
    """
    This function will return number of new features to assess the performance(at away only!) of a specific team in the past.
    Input:
        season: the season of match
        H_name: the name of home team of match
        A_name: the name of away team of match
        find_name: the name of team that want to search(can only be H_name or A_name)
        x: number of past matches of 'find_name' team
    Output:
        1.sum of goals of 'find_name' team in past x matches
        2.average goals of 'find_name' team in past x matches
        3.sum of goal difference of 'find_name' team in past x matches
        4.average goal difference of 'find_name' team in past x matches
        5.sum of yellow cards of 'find_name team' got in past x matches
        6.average of yellow cards of 'find_name team' got in past x matches
        7.sum of points that 'find_name team' obtained in past x matches
    """
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
        else:
            continue
        count += 1
        if count == x:
            return sum(goals) , sum(goals)/x, sum(goal_difference), sum(goal_difference)/x, sum(yellow_card), sum(yellow_card)/x, sum(points)
            break
    if count == 0:
        return None

def meetingrecord(season, H_name, A_name, findname, x):
    """
    This function will return two new features of past meeting records between home and away teams.
    Input:
        season: the season of match
        H_name: the name of home team of match
        A_name: the name of away team of match
        find_name: the name of team that want to search(can only be H_name or A_name)
        x: number of meeting records.
    Output:
        1.sum of goal difference of 'find_name' team in past x meeting records
        2.average goal difference of 'find_name' team in past x meeting record
    """
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

def countstreak(nums):
    """
    This function counts the maximum number of streaks in a sequence of number.
    Input:
        a sequence of number
    Output:
        1.maximum streaks of -1(losses)
        2.maximun streaks of 0(draws)
        3.maximum streaks of 1(wins)
    """
    res = {-1:0, 0:0, 1:0}
    l = 1
    previous = nums[0]
    for i in range(l, len(nums)):
        if nums[i] == previous:
            l += 1
        else:
            l = 1
        res[nums[i]] = max(res[nums[i]], l)
        previous = nums[i]
    return res[1], res[0], res[-1]

def streak(season, H_name, A_name, find_name, x):
    """
    This function returns maximum number of streaks(wins, draws and losses) of 'find_name' team in past x matches.
    Input:
        season: the season of match
        H_name: the name of home team of match
        A_name: the name of away team of match
        find_name: the name of team that want to search(can only be H_name or A_name)
        x: number of past matches of 'find_name' team
    Output:
        1. maximum streaks of wins in past x matches that achieved by 'find_name' team
        2. maximum streaks of draws in past x matches that achieved by 'find_name' team
        3. maximum streaks of losses in past x matches that achieved by 'find_name' team
    """
    loc = df[(df['Season'] == season) & (df['H'] == H_name) & (df['A'] == A_name)]
    if loc.size == 0:
        return None
    index = loc.index.tolist()[0]
    count = 0
    past = [] ## store win/draw/loss records of past x games of 'findname' team
    for i in range(index + 1, len(df)):
        if df.iloc[i]['A'] == find_name:
            if df.iloc[i]['A Goals'] - df.iloc[i]['H Goals'] > 0:
                past.append(1)# 1 stand for win
            elif df.iloc[i]['A Goals'] - df.iloc[i]['H Goals'] == 0:
                past.append(0)# 0 stand for draw
            else:
                past.append(-1)# -1 stand for draw
        elif df.iloc[i]['H'] == find_name:
            if df.iloc[i]['H Goals'] - df.iloc[i]['A Goals'] > 0:
                past.append(1)# 1 stand for win
            elif df.iloc[i]['H Goals'] - df.iloc[i]['A Goals'] == 0:
                past.append(0)# 0 stand for draw
            else:
                past.append(-1)# -1 stand for draw
        else:
            continue
        count += 1
        if count == x:
            return countstreak(past)
            break
    if count == 0:
        return None

# Dictionaries to store new features
result_H_past5 = {}
result_A_past5 = {}
result_H_past20 = {}
result_A_past20 = {}
Meeting_record_A_3 = {}
Meeting_record_H_3 = {}
result_H_at_home5 = {}
result_A_at_away5 = {}
streak_A_past5 = {}
streak_H_past5 = {}

season_cutoff = [199, 579, 959, 1219, 1449, 1599, 1979, 2359, 2739, 3119, 3499, 3879, 4259, 4639] # no. of rows which a new season start
season = 23

for i in range(0, len(df)):# Looping row by row to create new features
    for j in season_cutoff:
        if i == j:
            season -= 1
    Hteam = df.iloc[i,2]
    Ateam = df.iloc[i,3]
    result_H_past5[i] = findpast(season, Hteam, Ateam, Hteam, 5)
    result_A_past5[i] = findpast(season, Hteam, Ateam, Ateam, 5)
    result_H_past20[i] = findpast(season, Hteam, Ateam, Hteam, 20)
    result_A_past20[i] = findpast(season, Hteam, Ateam, Ateam, 20)
    result_H_at_home5[i] = findpast_homeonly(season, Hteam, Ateam, Hteam ,5)
    result_A_at_away5[i] = findpast_awayonly(season, Hteam, Ateam, Ateam ,5)
    Meeting_record_A_3[i] = meetingrecord(season, Hteam, Ateam, Ateam, 3)
    Meeting_record_H_3[i] = meetingrecord(season, Hteam, Ateam, Hteam, 3)
    streak_H_past5[i] = streak(season, Hteam, Ateam, Hteam, 5)
    streak_A_past5[i] = streak(season, Hteam, Ateam, Ateam, 5)


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
W_streaks = []
D_streaks = []
L_streaks = []
lists3 = [W_streaks,D_streaks,L_streaks]


# Load results from dictionary to lists
def load_results(lists, result):
    for lst in lists:
        lst.clear()  # clear them before use

    for i in range(len(result)):  # Iterate each element in dictionary
        if result[i] is None:
            for lst in lists:
                lst.append(None)
        else:
            for j, lst in enumerate(lists):
                lst.append(result[i][j])


load_results(lists, result_H_past5)
df = df.assign(H_goals5=goals, H_avg_goals5=avg_goals, H_goal_difference5=goal_difference,
               H_avg_goal_difference5=avg_goal_difference, H_yellow_card5=yellow_card,
               H_avg_yellow_card5=avg_yellow_card, H_points5=points)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists, result_A_past5)
df = df.assign(A_goals5=goals, A_avg_goals5=avg_goals, A_goal_difference5=goal_difference,
               A_avg_goal_difference5=avg_goal_difference, A_yellow_card5=yellow_card,
               A_avg_yellow_card5=avg_yellow_card, A_points5=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists, result_H_past20)
df = df.assign(H_goals20=goals, H_avg_goals20=avg_goals, H_goal_difference20=goal_difference,
               H_avg_goal_difference20=avg_goal_difference, H_yellow_card20=yellow_card,
               H_avg_yellow_card20=avg_yellow_card, H_points20=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists, result_A_past20)
df = df.assign(A_goals20=goals, A_avg_goals20=avg_goals, A_goal_difference20=goal_difference,
               A_avg_goal_difference20=avg_goal_difference, A_yellow_card20=yellow_card,
               A_avg_yellow_card20=avg_yellow_card, A_points20=points)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists2, Meeting_record_A_3)
df = df.assign(A_goal_difference3 = past_goal_difference, A_avg_goal_difference3 = avg_past_goal_difference)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists2, Meeting_record_H_3)
df = df.assign(H_goal_difference3 = past_goal_difference, H_avg_goal_difference3 = avg_past_goal_difference)
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists, result_H_at_home5)
df = df.assign(H_home_goals5=goals, H_home_avg_goals5=avg_goals, H_home_goal_difference5=goal_difference,
               H_home_avg_goal_difference5=avg_goal_difference, H_home_yellow_card5=yellow_card,
               H_home_avg_yellow_card5=avg_yellow_card, H_home_points5=points)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists, result_A_at_away5)
df = df.assign(A_away_goals5=goals, A_away_avg_goals5=avg_goals, A_away_goal_difference5=goal_difference,
               A_away_avg_goal_difference5=avg_goal_difference, A_away_yellow_card5=yellow_card,
               A_away_avg_yellow_card5=avg_yellow_card, A_away_points5=points)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists3, streak_H_past5)
df = df.assign(H_W_streaks=W_streaks, H_D_streaks=D_streaks, H_L_streaks=L_streaks)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)

load_results(lists3, streak_A_past5)
df = df.assign(A_W_streaks=W_streaks, A_D_streaks=D_streaks, A_L_streaks=L_streaks)  # Write list to csv file
df.to_csv('allSeasons - PremData - complete data - edited.csv', index=False)