import pandas as pd
import numpy as np
import pytest
import sys
sys.path.append('../src')
from preprocessing_helpers import (
    get_goals_scored, 
    get_goals_conceded, 
    get_gss, 
    get_points, 
    get_cuml_points, 
    get_matchres,
    get_agg_points,
    get_form,
    add_form,
    add_form_df
)

@pytest.fixture
def example_playing_stat():
    data = {'HomeTeam': ['Arsenal', 'Chelsea', 'Man Utd'],
            'AwayTeam': ['Chelsea', 'Man Utd', 'Arsenal'],
            'FTHG': [1, 2, 0],
            'FTAG': [2, 1, 2]}
    return pd.DataFrame(data)

@pytest.fixture
def example_playing_stat_2():    
    playing_stat = pd.DataFrame({
        'HomeTeam': ['Team A', 'Team B', 'Team C'],
        'AwayTeam': ['Team B', 'Team C', 'Team A'],
        'FTR': ['H', 'D', 'A']
    })
    return playing_stat

def test_get_goals_scored(example_playing_stat):
    expected_output = pd.DataFrame({
        'Arsenal': [1, 3, 0],
        'Chelsea': [2, 4, 0],
        'Man Utd': [1, 1, 0]
    }, index=[1, 2, 0]).T
    assert get_goals_scored(example_playing_stat).equals(expected_output)
    print(get_goals_scored(example_playing_stat))
    print(expected_output)

def test_get_goals_conceded(example_playing_stat):
    expected_output = pd.DataFrame({
        'Arsenal': [2, 2, 0],
        'Chelsea': [1, 2, 0],
        'Man Utd': [2, 4, 0]
    }, index=[1, 2, 0]).T
    assert get_goals_conceded(example_playing_stat).equals(expected_output)

def test_get_gss(example_playing_stat):
    expected_output = pd.DataFrame({
        'HomeTeam': ['Arsenal', 'Chelsea', 'Man Utd'],
        'AwayTeam': ['Chelsea', 'Man Utd', 'Arsenal'],
        'FTHG': [1, 2, 0],
        'FTAG': [2, 1, 2],
        'HTGS': [0, 0, 0],
        'ATGS': [0, 0, 0],
        'HTGC': [0, 0, 0],
        'ATGC': [0, 0, 0]
    })
    assert get_gss(example_playing_stat).equals(expected_output)

def test_get_points():
    assert get_points('W') == 3
    assert get_points('D') == 1
    assert get_points('L') == 0

def test_get_cuml_points():
    matchres = pd.DataFrame({
        'Arsenal': ['W', 'D', 'L'],
        'Chelsea': ['D', 'W', 'W'],
        'Man Utd': ['L', 'L', 'D']
    }, index=[1, 2, 3]).T
    expected_output = pd.DataFrame({
        0: [0, 0, 0],
        1: [3, 1, 0],
        2: [4, 4, 0],
        3: [4, 7, 1]
    }, index=['Arsenal', 'Chelsea', 'Man Utd'])
    assert get_cuml_points(matchres, 4, r=3).equals(expected_output)   

def test_get_matchres(example_playing_stat_2):
    expected_result = pd.DataFrame({
        'Team A': ['W', 'W'],
        'Team B': ['L', 'D'],
        'Team C': ['D', 'L']
    }, index = [1,2]).T
    result = get_matchres(example_playing_stat_2, 3, 3)
    pd.testing.assert_frame_equal(result, expected_result)

def test_get_agg_points(example_playing_stat_2):
    expected_result = pd.DataFrame({
        'HomeTeam': ['Team A', 'Team B', 'Team C'],
        'AwayTeam': ['Team B', 'Team C', 'Team A'],
        'FTR': ['H', 'D', 'A'],
        'HTP': [0, 0, 0],
        'ATP': [0, 0, 0]
    })
    result = get_agg_points(example_playing_stat_2, r=3)
    pd.testing.assert_frame_equal(result, expected_result)

def test_get_form(example_playing_stat_2):
    expected_result = pd.DataFrame({
        'Team A': ['W', 'W'],
        'Team B': ['L', 'D'],
        'Team C': ['D', 'L']
    }, index = [1,2]).T
    result = get_form(example_playing_stat_2, 3)
    pd.testing.assert_frame_equal(result, expected_result)

    
    