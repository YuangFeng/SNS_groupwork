import pandas as pd

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
    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')  # read data
    index = 0
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
    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')
    index = 0
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
    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')
    index = 0
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
    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')
    index = 0
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
    df = pd.read_csv('/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/allSeasons - PremData - complete data.csv')
    index = 0
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
                past.append(-1)# -1 stand for loss
        else:
            continue
        count += 1
        if count == x:
            return countstreak(past)
            break
    if count == 0:
        return None