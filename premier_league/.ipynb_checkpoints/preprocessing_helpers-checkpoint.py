import pandas as pd
import numpy as np
from datetime import datetime as dt
import itertools
import warnings
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import make_scorer, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder


def get_goals_scored(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the number of goals scored by each team over the 
    entire dataset.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics

    Returns:
    - big_df (DataFrame): Pandas dataframe with team names as rows and columns 
    as the number of goals scored by the team
    """
    # Create a dictionary with team names as keys
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

    big_df = pd.DataFrame()
    for team in teams.keys():
        mini_df = pd.DataFrame(teams[team]).T
        mini_df.index = [team]
        mini_df.columns = (mini_df.columns) + 1
        big_df = pd.concat([big_df, mini_df])
    big_df = big_df.fillna(0)
    big_df[0] = 0    
    
    for i in range(2,int((big_df.shape[1]) )):
        big_df[i] = big_df[i] + big_df[i-1]
    return big_df

def get_goals_conceded(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the number of goals conceded by each team over the entire 
    dataset.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics

    Returns:
    - big_df (DataFrame): Pandas dataframe with team names as rows and columns
    as the number of goals conceded by the team
    """
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}
    
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    big_df = pd.DataFrame()
    for team in teams.keys():
        mini_df = pd.DataFrame(teams[team]).T
        mini_df.index = [team]
        mini_df.columns = (mini_df.columns) + 1
        big_df = pd.concat([big_df, mini_df])
    big_df = big_df.fillna(0)
    big_df[0] = 0    
    
    for i in range(2,int((big_df.shape[1]) )):
        big_df[i] = big_df[i] + big_df[i-1]
    return big_df

def get_gss(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the number of goals scored and conceded by each team in each 
    match.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics

    Returns:
    - playing_stat (DataFrame): Pandas dataframe with columns for the number 
    of goals scored and conceded by the home team (HTGS, HTGC) and away team
    (ATGS, ATGC)
    """
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(playing_stat.shape[0]):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat


def get_points(
        result: str
    ) -> int:
    """
    Calculates the number of points earned by a team in a single match.

    Parameters:
    - result (str): String representing the match result, where 'W' 
    indicates a win, 'D' indicates a draw and 'L' indicates a loss.

    Returns:
    - points (int): Integer representing the number of points earned by the team
    """
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
    

def get_cuml_points(
        matchres: pd.DataFrame, 
        games_1: int,
        r: int = 20
    ) -> pd.DataFrame:
    """
    Calculates the cumulative number of points earned by each team 
    up to a given match.

    Parameters:
    - matchres (DataFrame): Pandas dataframe containing the match results
    - games_1 (int): Integer representing the number of games played
    per team plus one.

    Returns:
    - matchres_points (DataFrame): Pandas dataframe with columns representing
    each match and rows representing each team, with the value of each cell
    representing the number of points earned by the team up to that match.
    """
    matchres_points = matchres.applymap(get_points)
    for i in range(2,games_1):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
        
    matchres_points.insert(column =0, 
                           loc = 0, 
                           value = [0*i for i in range(r)])
    return matchres_points

def get_matchres(
        playing_stat: pd.DataFrame, 
        games_1: int, 
        total_games: int
    ) -> pd.DataFrame:
    """
    Calculates the match results for each team over the entire dataset.

    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics
    - games_1 (int): Integer representing the number of games played per 
    team plus one
    - total_games (int): Integer representing the total number of games in the
    dataset

    Returns:
    - big_df (DataFrame): Pandas dataframe with team names as rows and columns
    as the match results ('W', 'D' or 'L') for each match.
    """
    teams = {team: [] for team in playing_stat['HomeTeam'].unique()}

    # the value corresponding to keys is a list containing the match result
    for i in range(total_games):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    
    big_df = pd.DataFrame()
    for team in teams.keys():
        mini_df = pd.DataFrame(teams[team]).T
        mini_df.index = [team]
        mini_df.columns = (mini_df.columns) + 1
        big_df = pd.concat([big_df, mini_df])
    big_df = big_df.fillna('L')

    return big_df

   
def get_agg_points(
        playing_stat: pd.DataFrame,
        r:int = 20
    ) -> pd.DataFrame:
    """
    Calculates the cumulative number of points earned by each team up to a
    given match.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match 
    statistics.
    - r: Number of games in a season.
    
    Returns:
    - playing_stat (DataFrame): Pandas dataframe with additional columns 
    representing the cumulative number of points earned by the home team 
    (HTP) and away team (ATP) up to each match.
    """
    games_1 = int((playing_stat.shape[0] / 10) + 1 )
    total_games = len(playing_stat)
    matchres = get_matchres(playing_stat, games_1, total_games)
    cum_pts = get_cuml_points(matchres, games_1, r = r)
    HTP = []
    ATP = []
    j = 0
    for i in range(playing_stat.shape[0]):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat 


def get_form(
    playing_stat: pd.DataFrame,
    num: int
) -> pd.DataFrame:
    """
    Calculates the current form of each team based on their match results up 
    to a given match.

    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics
    - num (int): Integer representing the number of previous matches to consider
    when calculating form
    
    Returns:
    - form_final (DataFrame): Pandas dataframe with columns representing each 
    match and rows representing each team, with the value of each cell 
    representing the form of the team up to that match.
    """
    games_1 = int((playing_stat.shape[0] / 10) + 1 )
    total_games = len(playing_stat)

    form = get_matchres(playing_stat, games_1, total_games)
    form_final = form.copy()
    for i in range(num,int((playing_stat.shape[0] / 10) + 1 )):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final

def add_form(
        playing_stat: pd.DataFrame,
        num: int
    ) -> pd.DataFrame:
    """
    Adds columns to the playing statistics dataframe representing the current
    form of each team up to a given match.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match statistics
    - num (int): Integer representing the number of previous matches to consider
    when calculating form
    
    Returns:
    - playing_stat (DataFrame): Pandas dataframe with additional columns 
    representing the current form of each team up to each match.
    """
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num*10),playing_stat.shape[0]):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]              
        h.append(past[num-1])               
        
        past = form.loc[at][j]              
        a.append(past[num-1])               
        
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    return playing_stat

def add_form_df(
        playing_statistics: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds columns to the playing statistics dataframe representing the 
    current form of each team up to a given number of previous matches.
    
    Parameters:
    - playing_statistics (DataFrame): Pandas dataframe containing the match
    statistics
    
    Returns:
    - playing_statistics (DataFrame): Pandas dataframe with additional columns
    representing the current form of each team up to each of 5 previous 
    matches.
    """
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    playing_statistics = add_form(playing_statistics,4)
    playing_statistics = add_form(playing_statistics,5)
    return playing_statistics    

def get_mw(
    playing_stat: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds a column to the playing statistics dataframe representing the match
    week number for each match.
    
    Parameters:
    - playing_stat (DataFrame): Pandas dataframe containing the match 
    statistics
    
    Returns:
    - playing_stat (DataFrame): Pandas dataframe with an additional column 
    representing the match week number for each match.
    """
    j = 1
    MatchWeek = []
    for i in range(playing_stat.shape[0]):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

def gss_all_seasons(
        playing_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the number of goals scored and conceded by each team in each
    match for all seasons.
    
    Parameters:
    - playing_df (DataFrame): Pandas dataframe containing the match statistics 
    for all seasons.
    
    Returns:
    - gs_df_full (DataFrame): Pandas dataframe with columns for the number of
    goals scored and conceded by the home team (HTGS, HTGC) and away team 
    (ATGS, ATGC) for each match, for all seasons.
    """
    gs_df_full = pd.DataFrame()
    for season in sorted(playing_df['season'].unique()):
        gs_df = get_gss(playing_df[playing_df['season'] == season])
        gs_df_full = pd.concat([gs_df_full, gs_df])
    return gs_df_full

def points_all_seasons(
        playing_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the cumulative number of points earned by each team up to a
    given match for all seasons.
    
    Parameters:
    - playing_df (DataFrame): Pandas dataframe containing the match statistics 
    for all seasons.
    
    Returns:
    - points_df_full (DataFrame): Pandas dataframe with columns representing 
    each match and rows representing each team, with the value of each cell
    representing the number of points earned by the team up to that match, 
    for all seasons.
    """
    points_df_full = pd.DataFrame()
    for season in sorted(playing_df['season'].unique()):
        points_df = get_agg_points(playing_df[playing_df['season'] == season])
        points_df_full = pd.concat([points_df_full, points_df])
    return points_df_full
    
def form_all_seasons(
        playing_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Returns a dataframe with the form for each team over the previous 5 games
    for all seasons in the input dataframe.

    Parameters:
    - playing_df (pd.DataFrame): A pandas dataframe containing all of the
        Premier League data.

    Returns:
    - pd.DataFrame: A pandas dataframe containing the form for each team 
        over the previous 5 games for all seasons in the input dataframe.
    """
    form_df_full = pd.DataFrame()
    for season in sorted(playing_df['season'].unique()):
        form_df = add_form_df(playing_df[playing_df['season'] == season])
        form_df_full = pd.concat([form_df_full, form_df])
    return form_df_full

def mw_all_season(
        playing_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Returns a dataframe with the match week for each game in each season in
    the input dataframe.

    Parameters:
    - playing_df (pd.DataFrame): A pandas dataframe containing all of the
        Premier League data.

    Returns:
    - pd.DataFrame: A pandas dataframe containing the match week for each 
        game in each season in the input dataframe.
    """
    match_week_full = pd.DataFrame()
    for season in sorted(playing_df['season'].unique()):
        match_week = get_mw(playing_df[playing_df['season'] == season])
        match_week_full = pd.concat([match_week_full, match_week])
    return match_week_full


def merge_data(
    gs_df_full: pd.DataFrame, 
    points_df_full: pd.DataFrame, 
    form_df_full: pd.DataFrame,
    match_week_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges four dataframes (goals scored, points, form, and match week) 
    into a single dataframe.

    Parameters:
    - gs_df_full (pd.DataFrame): A pandas dataframe containing the goals
        scored for each team in each game.
    - points_df_full (pd.DataFrame): A pandas dataframe containing the 
        cumulative points for each team over the season.
    - form_df_full (pd.DataFrame): A pandas dataframe containing the form
        for each team over the previous 5 games.
    - match_week_full (pd.DataFrame): A pandas dataframe containing the
        match week for each game.

    Returns:
    - pd.DataFrame: A pandas dataframe with merged data.
    """
    keys = ['season', 'Date', 'HomeTeam', 'AwayTeam','FTHG', 'FTAG', 'FTR'] 
    playing_df = pd.merge(gs_df_full, points_df_full, on=keys )
    playing_df = pd.merge(playing_df, form_df_full, on=keys)
    playing_stat = pd.merge(playing_df, match_week_full, on=keys)
    return playing_stat


def get_form_points(
        string: str
    ) -> int:
    """
    Given a string of letters representing the results of the previous 5
    games, returns the total number of points
    earned by the team in those games.

    Parameters:
    - string (str): A string of letters representing the results of the 
        previous 5 games.

    Returns:
    - int: The total number of points earned by the team in the previous 
        5 games.
    """
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

def add_form_cols(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds columns to the input dataframe containing the total number of 
    points earned by each team over the previous 5 games and a string 
    representation of the results of the previous 5 games.

    Parameters:
    - playing_stat (pd.DataFrame): A pandas dataframe containing the
        Premier League data.

    Returns:
    - pd.DataFrame: A pandas dataframe with additional columns for the 
        form of each team.
    """
    playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat[
        'HM2'] + playing_stat['HM3'] + playing_stat[
        'HM4'] + playing_stat['HM5']
    playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat[
        'AM2'] + playing_stat['AM3'] + playing_stat[
        'AM4'] + playing_stat['AM5']
    playing_stat['HTFormPts'] = playing_stat[
        'HTFormPtsStr'].apply(get_form_points)
    playing_stat['ATFormPts'] = playing_stat[
        'ATFormPtsStr'].apply(get_form_points)
    return playing_stat

def get_3game_ws(
        string: str
    ) -> int:
    """
    Returns 1 if a team has won their last three games, 0 otherwise.
    
    Parameters:
    - string (str): A string containing the results of a team's previous games.
    
    Returns:
    - int: 1 if the team has won their last three games, 0 otherwise.
    """
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(
        string: str
    ) -> int:
    """
    Returns 1 if a team has won their last five games, 0 otherwise.
    
    Parameters:
    - string (str): A string containing the results of a team's previous games.
    
    Returns:
    - int: 1 if the team has won their last five games, 0 otherwise.
    """
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(
        string: str
    ) -> int:
    """
    Returns 1 if a team has lost their last three games, 0 otherwise.
    
    Parameters:
    - string (str): A string containing the results of a team's previous games.
    
    Returns:
    - int: 1 if the team has lost their last three games, 0 otherwise.
    """
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(
        string: str
    ) -> int:
    """
    Returns 1 if a team has lost their last five games, 0 otherwise.
    
    Parameters:
    - string (str): A string containing the results of a team's previous games.
    
    Returns:
    - int: 1 if the team has lost their last five games, 0 otherwise.
    """
    if string == 'LLLLL':
        return 1
    else:
        return 0

def add_win_streaks(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds columns to `playing_stat` indicating whether each team has won or lost three or five games in a row.
    
    Parameters:
    - playing_stat (pandas.DataFrame): A DataFrame containing the playing statistics for each team.
    
    Returns:
    - pandas.DataFrame: A copy of `playing_stat` with new columns indicating win/loss streaks for each team.
    """
    playing_stat['HTWinStreak3'] = playing_stat[
        'HTFormPtsStr'].apply(get_3game_ws)
    playing_stat['HTWinStreak5'] = playing_stat[
        'HTFormPtsStr'].apply(get_5game_ws)
    playing_stat['HTLossStreak3'] = playing_stat[
        'HTFormPtsStr'].apply(get_3game_ls)
    playing_stat['HTLossStreak5'] = playing_stat[
        'HTFormPtsStr'].apply(get_5game_ls)
    
    playing_stat['ATWinStreak3'] = playing_stat[
        'ATFormPtsStr'].apply(get_3game_ws)
    playing_stat['ATWinStreak5'] = playing_stat[
        'ATFormPtsStr'].apply(get_5game_ws)
    playing_stat['ATLossStreak3'] = playing_stat[
        'ATFormPtsStr'].apply(get_3game_ls)
    playing_stat['ATLossStreak5'] = playing_stat[
        'ATFormPtsStr'].apply(get_5game_ls)
    return playing_stat

def add_goal_difference(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates and adds the goal difference columns to the playing_stat DataFrame.

    Parameters:
    - playing_stat (pd.DataFrame): The DataFrame containing playing statistics.

    Returns:
    - pd.DataFrame: The playing_stat DataFrame with added 'HTGD' and 'ATGD' columns.
    """
    playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
    playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']
    return playing_stat


def add_points_diff(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates and adds the points difference columns to the playing_stat DataFrame.

    Parameters:
    - playing_stat (pd.DataFrame): The DataFrame containing playing statistics.

    Returns:
    - pd.DataFrame: The playing_stat DataFrame with added 'DiffPts' and 'DiffFormPts' columns.
    """
    playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
    playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']
    return playing_stat


def league_position(
        playing_stat: pd.DataFrame, 
        season: int, 
        mw: int
    ) -> pd.DataFrame:
    """
    Computes the league position for each team in a given season and matchweek.

    Parameters:
    - playing_stat (pd.DataFrame): The DataFrame containing playing statistics.
    - season (int): The season for which the league position is calculated.
    - mw (int): The matchweek for which the league position is calculated.

    Returns:
    - pd.DataFrame: The league position DataFrame with columns 'season', 'MW', 'Team',
                      'LOC' (Home or Away), and 'LeaguePosition'.
    """
    league_pos = playing_stat[(playing_stat['season'] == season) & (playing_stat['MW'] == mw)]
    
    league_pos_df = pd.concat([
        pd.DataFrame({
            'Team': league_pos['HomeTeam'],
            'Points': league_pos['HTP'],
            'GD': league_pos['HTGD'],
            'MW': league_pos['MW'],
            'LOC': 'Home',
            'season': league_pos['season']
        }),
        pd.DataFrame({
            'Team': league_pos['AwayTeam'],
            'Points': league_pos['ATP'],
            'GD': league_pos['ATGD'],
            'MW': league_pos['MW'],
            'LOC': 'Away',
            'season': league_pos['season']
        })
    ])
    
    league_pos_df = league_pos_df.sort_values(by=[
        'Points', 'GD', 'Team']).drop_duplicates(subset='Team')
    league_pos_df['LeaguePosition'] = league_pos_df[
        'Points'].rank(method='first', ascending=False).astype(int)
    league_pos_df = league_pos_df[[
        'season', 'MW', 'Team', 'LOC', 'LeaguePosition']]
    
    home_df = league_pos_df[league_pos_df[
        'LOC'] == 'Home'].drop('LOC', axis=1)
    away_df = league_pos_df[league_pos_df[
        'LOC'] == 'Away'].drop('LOC', axis=1)
    key = ['season', 'MW']
    
    league_pos = pd.merge(league_pos, home_df, 
                          left_on=key + ['HomeTeam'], 
                          right_on=key + ['Team'], how='left').drop(
        'Team', axis=1).rename(columns={'LeaguePosition': 'HomeLeaguePosition'})
    league_pos = pd.merge(league_pos, away_df, left_on=key + [
        'AwayTeam'], right_on=key + ['Team'], how='left').drop(
        'Team', axis=1).rename(columns={
        'LeaguePosition': 'AwayLeaguePosition'})
    return league_pos


def league_pos_all_seasons(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Computes the league position for all seasons and matchweeks in the playing_stat DataFrame.

    Parameters:
    - playing_stat (pd.DataFrame): The DataFrame containing playing statistics.

    Returns:
    - pd.DataFrame: The final league position DataFrame with columns 'season', 'MW', 'Team',
                      'LOC' (Home or Away), 'HomeLeaguePosition', 'AwayLeaguePosition',
                      and 'LeaguePositionDiff'.
    """
    league_pos_final = pd.DataFrame()
    for season in sorted(playing_stat['season'].unique()):
        for MW in sorted(playing_stat['MW'].unique()):
            league_pos_mini = league_position(playing_stat, season, MW)
            league_pos_final = pd.concat([league_pos_final, league_pos_mini])
            
    league_pos_final['AwayLeaguePosition'] = league_pos_final.groupby([
        'season', 'AwayTeam'])['AwayLeaguePosition'].ffill().fillna(0).astype(int)
    league_pos_final[
        'LeaguePositionDiff'] = league_pos_final[
        'HomeLeaguePosition'] - league_pos_final['AwayLeaguePosition']
    return league_pos_final


def scale_points_mw(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Scales the points-based columns in the playing_stat DataFrame by matchweek.

    Parameters:
    - playing_stat (pd.DataFrame): The DataFrame containing playing statistics.

    Returns:
    - pd.DataFrame: The playing_stat DataFrame with scaled columns 'HTGD', 'ATGD',
                      'DiffPts', 'DiffFormPts', 'HTP', and 'ATP'.
    """
    cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
    playing_stat['MW'] = playing_stat['MW'].astype(float)
    
    for col in cols:
        playing_stat[col] = playing_stat[col] / playing_stat['MW']
    
    return playing_stat


def convert_to_date(
        date: str
    ) -> pd.Timestamp:
    """
    Converts a date string to a pandas datetime object.

    Parameters:
    - date (str): The date string to convert.

    Returns:
    - pd.Timestamp: The pandas datetime object representing the converted date.

    Raises:
    - ValueError: If the date format is not recognized.
    """
    try:
        if len(date) < 10:
            pd_date = pd.to_datetime(date, format='%d/%m/%y')
        else:
            pd_date = pd.to_datetime(date, format='%d/%m/%Y')
        return pd_date
    except ValueError:
        raise ValueError(f"Date format not recognized for '{date}'")


def add_date_features(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the goal difference for each team in the given playing_stat
    DataFrame.
    
    Parameters:
    - playing_stat (DataFrame): A pandas DataFrame with columns for the home 
    and away team goals scored and conceded.
    
    Returns:
    - DataFrame: A new pandas DataFrame with additional columns for the home 
    and away team goal difference.
    """
    playing_stat['pd_date'] =  playing_stat[
        'Date'].apply(convert_to_date)
    playing_stat['MatchDayDay'] = playing_stat[
        'pd_date'].dt.strftime("%A")
    playing_stat['MatchDayMonth'] = playing_stat[
        'pd_date'].dt.strftime("%B")
    playing_stat['MatchDayDate'] = playing_stat[
        'pd_date'].dt.strftime("%d")
    return playing_stat
    
def convert_to_string(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculates the difference in points and form points between the home 
    and away team for each game in the given playing_stat DataFrame.
    
    Parameters:
    - playing_stat (DataFrame): A pandas DataFrame with columns for the home
    and away team points and form points.
    
    Returns:
    - DataFrame: A new pandas DataFrame with additional columns for the 
    difference in points and form points.
    """
    playing_stat['HM1'] = playing_stat['HM1'].astype('str')
    playing_stat['HM2'] = playing_stat['HM2'].astype('str')
    playing_stat['HM3'] = playing_stat['HM3'].astype('str')
    playing_stat['AM1'] = playing_stat['AM1'].astype('str')
    playing_stat['AM2'] = playing_stat['AM2'].astype('str')
    playing_stat['AM3'] = playing_stat['AM3'].astype('str')
    return playing_stat

def drop_cols(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Drops unneeded columns from the DataFrame.
    
    Parameters:
    - playing_stat (DataFrame): A pandas DataFrame containing premier league
    data.
    
    Returns:
    - DataFrame: A new pandas DataFrame with additional columns for the 
    difference in points and form points.
    """
    drop_cols = ['Date', 'FTHG', 'FTAG', 'HTFormPtsStr', 
                 'ATFormPtsStr', 'pd_date']
    for c in drop_cols:
        try:
            playing_stat = playing_stat.drop(c, axis=1)
        except KeyError as e:
            print(f'Drop cols: column {c} does not exist')
    return playing_stat



def convert_categorical(
        playing_stat: pd.DataFrame, 
        target: str
    ) -> pd.DataFrame:
    """
    Converts categorical variables in the dataset to numerical 
    values using the TargetEncoder. 
    
    Parameters:
    - playing_stat: Pandas DataFrame. The dataset containing the 
        categorical variables to be encoded. 
    - target: String. The name of the target variable in the dataset.
    
    Returns:
    - Pandas DataFrame. If the target variable is FTHG or FTAG, the 
    function returns a numeric dataset after encoding the categorical
    features. Otherwise, it returns the original dataset with 
    categorical variables encoded.
    """
    cat_cols = playing_stat.select_dtypes(
        'object').columns.to_list()
    if target=='FTR':
        cat_cols.remove(target)
    
    playing_stat[cat_cols] = playing_stat[cat_cols].astype('category')
    
    if target=='FTHG' or target=='FTAG':
        enc = TargetEncoder(cols=cat_cols,
                        min_samples_leaf=20, 
                        smoothing=10)
        enc = enc.fit_transform(X=playing_stat.drop(target, axis=1), 
                  y=playing_stat[target])
        numeric_dataset = enc.transform(X)
        return numeric_dataset
    else: 
        return playing_stat
    
def only_hw(
        string: str
    ) -> int:
    """
    Takes a string value and returns an integer based on the outcome 
    of the game. A value of 0 is returned for a home win, 1 for a draw, 
    and 2 for an away win.

    Parameters:
    - string: String. The outcome of a game (H for home win, D for draw,
    A for away win).
    
    Returns:
    - Integer. 0 for a home win, 1 for a draw, and 2 for an away win.
    """
    if string == 'H':
        return 0
    elif string == 'D':
        return 1
    else:
        return 2

def add_target(
        playing_stat: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds the target to the dataframe.

    Parameters:
    - playing_stat: Pandas DataFrame. The dataset containing the raw 
    game data.
    
    Returns:
    - Pandas DataFrame. The final dataset after adding the target
    """
    playing_stat['FTR'] = playing_stat['FTR'].apply(only_hw)
    return playing_stat

def run_pipeline(
        playing_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Runs the pipeline for generating the final dataset. 

    Parameters:
    - playing_df: Pandas DataFrame. The dataset containing the raw 
    game data.
    
    Returns:
    - Pandas DataFrame. The final dataset after running the entire 
    pipeline.
    """
    gs_df_full = gss_all_seasons(playing_df)
    points_df_full = points_all_seasons(playing_df)
    form_df_full = form_all_seasons(playing_df)
    match_week_full = mw_all_season(playing_df)
    playing_stat = merge_data(gs_df_full, points_df_full, 
                              form_df_full, match_week_full)
    playing_stat = add_form_cols(playing_stat)
    playing_stat = add_goal_difference(playing_stat)
    playing_stat = add_points_diff(playing_stat)
    
    playing_stat = league_pos_all_seasons(playing_stat)
    playing_stat = scale_points_mw(playing_stat)
    playing_stat = add_date_features(playing_stat)
    playing_stat = convert_to_string(playing_stat)
    playing_stat = drop_cols(playing_stat)
    playing_stat = convert_categorical(playing_stat, target='FTR')
    playing_stat = playing_stat.reset_index(drop=True)
    playing_stat[
        'MatchDayDate'] = playing_stat['MatchDayDate'].astype(int)
    playing_stat[
        'season'] = playing_stat['season'].astype(str).astype('category')
    playing_stat = add_target(playing_stat)
    return playing_stat

def add_season(
        data: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds a season column to the dataset by extracting the year from 
    the date column and determining the start and end years of the 
    season based on the month of the game.

    Parameters:
    - data: Pandas DataFrame. The dataset containing the date column.
    
    Returns:
    - Pandas DataFrame. The original dataset with the season column 
    added.
    """
    data['pd_date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['year'] = data['pd_date'].dt.strftime('%Y')
    data['last_year'] = (data['pd_date'].dt.strftime('%Y')).astype(int) - 1
    data['next_year'] = (data['pd_date'].dt.strftime('%Y')).astype(int) + 1
    data['month_num'] = data['pd_date'].dt.strftime('%m').astype(int)
    data['season_1'] = data[
        'last_year'].astype(str) + '-' + data['year'].str.slice(2,4) 
    data['season_2'] = data[
        'year'].astype(str) + '-' + data[
        'next_year'].astype(str).str.slice(2,4) 
    data['season'] = np.where(data[
        'month_num'] < 8, data['season_1'], data['season_2'])
    data = data.drop(['pd_date', 'year', 'last_year', 'next_year',
                        'month_num', 'season_1', 'season_2'],axis=1)
    return data
