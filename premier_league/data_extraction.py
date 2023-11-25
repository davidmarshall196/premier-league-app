
import pandas as pd
import os
import logging
try:
    from premier_league import (
        constants,
        s3_helpers,
        logger_config
    )
except ImportError:
    import constants
    import s3_helpers
    import logger_config
import importlib
import datetime
from datetime import datetime as dt
import numpy as np
from urllib.error import HTTPError

def generate_seasons_list(start, end):
    seasons_list = []
    for year in range(start, end):
        if year >= 2000:  # After Y2K
            season = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        else:  # Before Y2K
            season = f"{str(year)[-2:]}{str(year+1)[-2:]}".zfill(4)
        seasons_list.append(season)
    return seasons_list

def load_all_data(save_loc, cols_req):
    full_csv_list = []
    s_list = generate_seasons_list(1995, 2029)
    for s in s_list:
        full_csv_list.append(
            f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv')

    full_df = pd.DataFrame()
    for n, csv in enumerate(full_csv_list):
        logger_config.logger.info(f'Attempting csv {n+1} / {len(full_csv_list)}')
        logger_config.logger.info(f'Trying {csv}')
        try:
            data = pd.read_csv(csv)
            if csv[40:41] == '9':
                season = f'19{csv[40:42]}-{csv[42:44]}'
            else:
                season = f'20{csv[40:42]}-{csv[42:44]}'
            data['season'] = season
            data = data[cols_req]
            full_df = pd.concat([full_df, data])
        except Exception as e:
            try:
                data = pd.read_csv(csv,
                      encoding= 'unicode_escape',
                      usecols = ['Date', 'HomeTeam', 
                             'AwayTeam', 'FTHG', 'FTAG','FTR'])
                if csv[40:41] == '9':
                    season = f'19{csv[40:42]}-{csv[42:44]}'
                else:
                    season = f'20{csv[40:42]}-{csv[42:44]}'
                data['season'] = season
                data = data[cols_req]
                full_df = pd.concat([full_df, data])
            except Exception as e:
                print(e)
    full_df = full_df.dropna()
    s3_helpers.save_data_s3(
        full_df,
        save_loc
    )
    return full_df

def extract_season():
    c_date = pd.to_datetime(
        datetime.datetime.now().strftime('%Y-%m-%d'), 
        format='%Y-%m-%d')
    current_year = c_date.strftime('%Y')
    last_year = int(c_date.strftime('%Y')) - 1
    next_year = int(c_date.strftime('%Y')) + 1
    month_num = int(c_date.strftime('%m'))
    season_1 = str(last_year)[2:4] + str(current_year)[2:4]
    season_2 = str(current_year)[2:4] + str(next_year)[2:4] 
    if month_num < 8:
        season = season_1
    else:
        season = season_2
    return season
   
def add_new_data(full_data, cols_req, save_loc,
                align_20=False):
    s = extract_season()
    s1 = '20' + s[0:2] + '-' + s[2:4]
    full_data_remove = full_data[full_data['season'] != s1]
    try:
        csv_name = f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv'
        logger_config.logger.info(
            f'Trying to read data from {csv_name}')
        new_data = pd.read_csv(csv_name)
        season = f'20{s[0:2]}-{s[2:4]}'
        new_data['season'] = season
        new_data = new_data[cols_req]
        if align_20:
            if new_data.shape[0] % 20 != 0:
                rm = new_data.shape[0] % 20
                new_data = new_data.head(new_data.shape[0] - rm) 
        full_data_remove = pd.concat([full_data_remove, new_data])
        full_data_remove = full_data_remove.drop_duplicates()
        full_data_remove = full_data_remove.dropna()
        s3_helpers.save_data_s3(
            full_data_remove,
            save_loc
        )
        logger_config.logger.info(
            f'Data saved at {save_loc}')
        return full_data_remove
    except HTTPError as e:
        logger_config.logger.error(
            f'No data found at {save_loc}')
        return full_data
    

def replace_team_names(input_data, 
                       column, 
                       replace_map=constants.TEAM_NAME_REPLACEMENTS):
    input_data[column] = input_data[column].replace(replace_map)
    return input_data
    

def get_fixture_times():
    s = extract_season()
    s1 = s[0:2]
    csv_name = f'https://fixturedownload.com/download/epl-20{s1}-UTC.csv'
    logger_config.logger.info(f'Attempting to read from {csv_name}')
    fix = pd.read_csv(csv_name)
    fix = fix[fix['Result'].isna()]
    if fix.shape[0] == 0:
        print('No fixtures. Attempting to get next season')
        s_new = str(int(s1) + 1)
        csv_name = f'https://fixturedownload.com/download/epl-20{s_new}-UTC.csv'
        logger_config.logger.info(f'Attempting to read from {csv_name}')
        fix = pd.read_csv(csv_name)
        fix = fix[fix['Result'].isna()]
        if fix.shape[0] > 0:
            logger_config.logger.info(f'Successfully read from {csv_name}')
        else:
            logger_config.logger.error(f'No data found at {csv_name}')
            raise ValueError('No fixtures found. Please check')
    fix['season'] = '20' + s[0:2] + '-' + s[2:4]
    fix.columns = ['Match Number', 'Round Number', 'Date', 
                   'Location', 'HomeTeam', 'AwayTeam', 
                   'Result', 'season']
    fix = replace_team_names(fix, 'HomeTeam')
    fix = replace_team_names(fix, 'AwayTeam')
    return fix

def extract_current_fixtures(transformed_data):
    fix = get_fixture_times()
    fix = fix.drop(['Result'], axis=1)
    transformed_data['season'] = transformed_data[
        'season'].replace('season_', '')
    current_fixtures = pd.merge(transformed_data, fix,
                                on = ['season', 'HomeTeam', 'AwayTeam'],
                                how='inner')
    assert current_fixtures.shape[0] > 0
    current_fixtures['Fixture Date'] = pd.to_datetime(
    current_fixtures['Date'].str.slice(0,10), format = '%d/%m/%Y')
    current_fixtures['Fixture Time'] = (current_fixtures[
        'Date'].str.slice(11,13).astype(
        int) + 1).astype(str) + current_fixtures[
        'Date'].str.slice(13,16)
    c_date = pd.to_datetime(
        datetime.datetime.now().strftime('%Y-%m-%d'), 
        format='%Y-%m-%d')
    current_fixtures = current_fixtures.sort_values(
        by = ['Fixture Date', 'Fixture Time'])
    current_fixtures = current_fixtures[
        current_fixtures['Fixture Date'] >= c_date]
    current_fixtures['Fixture'] = current_fixtures[
        'HomeTeam'] + ' v ' + current_fixtures['AwayTeam']
    current_fixtures = current_fixtures.head(12)
    return current_fixtures

def get_fixtures(full_data):
    s = extract_season()
    s1 = s[0:2]
    csv_name = f'https://fixturedownload.com/download/epl-20{s1}-UTC.csv'
    logger_config.logger.info(
        f'Loading data from {csv_name}')
    fix = pd.read_csv(csv_name)
    logger_config.logger.info(
        f'Loaded data from {csv_name}')
    fix = fix[fix['Result'].isna()]
    fix['season'] = '20' + s[0:2] + '-' + s[2:4]
    fix = fix[['season', 'Date', 'Home Team', 'Away Team']]
    fix.columns = ['season', 'Date', 'HomeTeam', 'AwayTeam']
    fix['Date'] = fix['Date'].str.slice(0,10)
    fix['FTHG'] = np.nan
    fix['FTAG'] = np.nan
    fix['FTR'] = np.nan
    full_data = pd.concat([full_data, fix])
    full_data = replace_team_names(full_data, 'HomeTeam')
    full_data = replace_team_names(full_data, 'AwayTeam')
    full_data['FTHG'] = full_data['FTHG'].fillna(0)
    full_data['FTHG'] = full_data['FTHG'].fillna(0)
    full_data['FTHG'] = full_data['FTHG'].fillna('D')
    full_data = full_data.drop_duplicates(
        subset=['season', 'Date', 'HomeTeam', 'AwayTeam'],
        keep='first'
    )
    return full_data
    

