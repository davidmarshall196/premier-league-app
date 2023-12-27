import pytest
import pandas as pd
import datetime
import numpy as np
from unittest.mock import patch
try:
    from premier_league import data_extraction
except ImportError:
    import data_extraction


def test_generate_seasons_list():
    # Test range completely before 2000
    assert data_extraction.generate_seasons_list(
        1998, 2000) == ['9899', '9900'], "Failed for range before 2000"

    # Test range including 2000 transition
    assert data_extraction.generate_seasons_list(
        1999, 2001) == ['9900', '0001'], "Failed for range including 2000"

    # Test range completely after 2000
    assert data_extraction.generate_seasons_list(
        2000, 2002) == ['0001', '0102'], "Failed for range after 2000"

    # Test range with a single year (should return an empty list)
    assert data_extraction.generate_seasons_list(
        2000, 2000) == [], "Failed for single year range"

    # Test range with start year greater than end year (should also return an empty list)
    assert data_extraction.generate_seasons_list(
        2002, 2000) == [], "Failed for start year greater than end year"

def test_replace_team_names(mocker):
    # Mock DataFrame and replacement map
    data = pd.DataFrame({
        'Team': ['Gunners', 'Spurs', 'Forest'],
        'Points': [68, 72, 86]
    })
    replace_map = {
        'Gunners': 'Arsenal',
        'Spurs': 'Tottenham',
        'Forest': 'Nottingham Forest'
    }

    # Constants
    mocker.patch('premier_league.constants.TEAM_NAME_REPLACEMENTS', replace_map)

    # Test replacement
    result = data_extraction.replace_team_names(data, 'Team', replace_map)
    expected = pd.DataFrame({
        'Team': ['Arsenal', 'Tottenham', 'Nottingham Forest'],
        'Points': [68, 72, 86]
    })
    pd.testing.assert_frame_equal(result, expected, "Failed to replace team names correctly")

    # Test with empty replacement map
    empty_replace_map = {}
    mocker.patch('premier_league.constants.TEAM_NAME_REPLACEMENTS', empty_replace_map)
    result_empty_map = data_extraction.replace_team_names(data, 'Team')
    pd.testing.assert_frame_equal(result_empty_map, data, "Failed")

@pytest.fixture
def mock_season():
    with patch('premier_league.data_extraction.extract_season', return_value='21-22'):
        yield

@pytest.fixture
def mock_csv():
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(data = {
            "Match Number": [0, 1, 2, 3],
            "Round Number": [1, 1, 1, 1],
            "Date": ["11/08/2023 19:00", "12/08/2023 12:00", "12/08/2023 14:00", "12/08/2023 14:00"],
            "Location": ["Turf Moor", "Emirates Stadium", "Vitality Stadium", "Amex Stadium"],
            "Home Team": ["Burnley", "Arsenal", "Bournemouth", "Brighton"],
            "Away Team": ["Man City", "Nottingham Forest", "West Ham", "Luton"],
            "Result": ["0 - 3", "2 - 1", "1 - 1", "4 - 1"]
        })
        yield mock_read_csv

def test_get_fixture_times_no_fixtures(mock_season, mock_csv):
    with pytest.raises(ValueError):
        data_extraction.get_fixture_times()

