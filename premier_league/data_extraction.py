import pandas as pd

try:
    from premier_league import constants, s3_helpers, logger_config
except ImportError:
    import constants
    import s3_helpers
    import logger_config
import datetime
import numpy as np
from urllib.error import HTTPError
from typing import Optional, List


def generate_seasons_list(start: int, end: int) -> list:
    """
    Generate a list of season codes within a specified range of years.

    Parameters:
    - start (int): The starting year.
    - end (int): The ending year (exclusive).

    Returns:
    - list: A list of season codes (e.g., '9900' for 1999-2000).

    Note:
    The function handles the transition from pre to post Y2K by checking the year.
    """
    seasons_list = []
    for year in range(start, end):
        if year >= 2000:  # After Y2K
            season = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        else:  # Before Y2K
            season = f"{str(year)[-2:]}{str(year+1)[-2:]}".zfill(4)
        seasons_list.append(season)
    return seasons_list


def load_all_data(save_loc: str, cols_req: List[str]) -> pd.DataFrame:
    """
    Load data from multiple seasons of Premier League matches and
    save it to a specified location.

    Parameters:
    - save_loc (str): The location where the data will be saved.
    - cols_req (List[str]): A list of column names required in the data.

    Returns:
    - pd.DataFrame: A DataFrame containing the combined data from all the seasons.

    Notes:
    - Utilises the `generate_seasons_list`
    function to determine the range of seasons.
    - Logs information during the data loading process.
    - Handles exceptions during data loading, attempting a secondary encoding if needed.
    - Saves the compiled DataFrame to the specified location using
    `s3_helpers.save_data_s3`.
    """
    full_csv_list = []
    s_list = generate_seasons_list(1995, 2029)
    for s in s_list:
        full_csv_list.append(f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv")

    full_df = pd.DataFrame()
    for n, csv in enumerate(full_csv_list):
        logger_config.logger.info(f"Attempting csv {n+1} / {len(full_csv_list)}")
        logger_config.logger.info(f"Trying {csv}")
        try:
            data = pd.read_csv(csv)
            if csv[40:41] == "9":
                season = f"19{csv[40:42]}-{csv[42:44]}"
            else:
                season = f"20{csv[40:42]}-{csv[42:44]}"
            data["season"] = season
            data = data[cols_req]
            full_df = pd.concat([full_df, data])
        except Exception:
            try:
                data = pd.read_csv(
                    csv,
                    encoding="unicode_escape",
                    usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                )
                if csv[40:41] == "9":
                    season = f"19{csv[40:42]}-{csv[42:44]}"
                else:
                    season = f"20{csv[40:42]}-{csv[42:44]}"
                data["season"] = season
                data = data[cols_req]
                full_df = pd.concat([full_df, data])
            except Exception as e:
                logger_config.logger.error(e)
    full_df = full_df.dropna()
    s3_helpers.save_data_s3(full_df, save_loc)
    return full_df


def extract_season() -> str:
    """
    Extract the current Premier League season based on
    the current date.

    Returns:
    - str: The current season in the format 'YY-YY'.

    Notes:
    - The season is considered to start in August and end in July
        of the following year.
    """
    c_date = pd.to_datetime(
        datetime.datetime.now().strftime("%Y-%m-%d"), format="%Y-%m-%d"
    )
    current_year = c_date.strftime("%Y")
    last_year = int(c_date.strftime("%Y")) - 1
    next_year = int(c_date.strftime("%Y")) + 1
    month_num = int(c_date.strftime("%m"))
    season_1 = str(last_year)[2:4] + str(current_year)[2:4]
    season_2 = str(current_year)[2:4] + str(next_year)[2:4]
    if month_num < 8:
        season = season_1
    else:
        season = season_2
    return season


def add_new_data(
    full_data: pd.DataFrame, cols_req: List[str], save_loc: str, align_20: bool = False
) -> pd.DataFrame:
    """
    Add new season data to an existing dataset and save the updated dataset.

    Parameters:
    - full_data (pd.DataFrame): The existing dataset.
    - cols_req (List[str]): A list of required column names.
    - save_loc (str): The location where the updated dataset will be saved.
    - align_20 (bool): If True, aligns the number of rows to a multiple
    of 20 by trimming the dataset. Defaults to False.

    Returns:
    - pd.DataFrame: The updated dataset including the new season data.

    Notes:
    - Uses the `extract_season` function to determine the current season.
    - Removes existing data of the current season before adding new data.
    - In case of HTTPError, logs an error and returns the original
    full_data without changes.
    - Saves the updated dataset using `s3_helpers.save_data_s3`.
    """
    s = extract_season()
    s1 = "20" + s[0:2] + "-" + s[2:4]
    full_data_remove = full_data[full_data["season"] != s1]
    try:
        csv_name = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
        logger_config.logger.info(f"Trying to read data from {csv_name}")
        new_data = pd.read_csv(csv_name)
        season = f"20{s[0:2]}-{s[2:4]}"
        new_data["season"] = season
        new_data = new_data[cols_req]
        if align_20:
            if new_data.shape[0] % 20 != 0:
                rm = new_data.shape[0] % 20
                new_data = new_data.head(new_data.shape[0] - rm)
        full_data_remove = pd.concat([full_data_remove, new_data])
        full_data_remove = full_data_remove.drop_duplicates()
        full_data_remove = full_data_remove.dropna()
        s3_helpers.save_data_s3(full_data_remove, save_loc)
        logger_config.logger.info(f"Data saved at {save_loc}")
        return full_data_remove
    except HTTPError:
        logger_config.logger.error(f"No data found at {save_loc}")
        return full_data


def replace_team_names(
    input_data: pd.DataFrame,
    column: str,
    replace_map: dict = constants.TEAM_NAME_REPLACEMENTS,
) -> pd.DataFrame:
    """
    Replace team names in a specified column of a DataFrame according
    to a given mapping.

    Parameters:
    - input_data (pd.DataFrame): The DataFrame containing team names.
    - column (str): The name of the column in which team names are to
    be replaced.
    - replace_map (dict): A dictionary mapping old team names to
    new ones. Defaults to constants.TEAM_NAME_REPLACEMENTS.

    Returns:
    - pd.DataFrame: The DataFrame with team names replaced in the
    specified column.

    Notes:
    - This function is useful for standardizing team names across
    different datasets.
    """
    input_data[column] = input_data[column].replace(replace_map)
    return input_data


def get_fixture_times() -> Optional[pd.DataFrame]:
    """
    Retrieve the upcoming fixture times for the current or next Premier
    League season.

    Returns:
    - Optional[pd.DataFrame]: A DataFrame containing the upcoming
    fixtures, or None if no fixtures are found.

    Notes:
    - Uses the `extract_season` function to determine the current season.
    - Attempts to retrieve fixtures for the next season if none are found
    for the current season.
    - Logs the process and any errors encountered.
    - Standardizes team names using the `replace_team_names` function.
    - Raises ValueError if no fixtures are found after attempting both
    the current and next season.
    """
    s = extract_season()
    s1 = s[0:2]
    csv_name = f"https://fixturedownload.com/download/epl-20{s1}-UTC.csv"
    logger_config.logger.info(f"Attempting to read from {csv_name}")
    fix = pd.read_csv(csv_name)
    fix = fix[fix["Result"].isna()]
    if fix.shape[0] == 0:
        print("No fixtures. Attempting to get next season")
        s_new = str(int(s1) + 1)
        csv_name = f"https://fixturedownload.com/download/epl-20{s_new}-UTC.csv"
        logger_config.logger.info(f"Attempting to read from {csv_name}")
        fix = pd.read_csv(csv_name)
        fix = fix[fix["Result"].isna()]
        if fix.shape[0] > 0:
            logger_config.logger.info(f"Successfully read from {csv_name}")
        else:
            logger_config.logger.error(f"No data found at {csv_name}")
            raise ValueError("No fixtures found. Please check")
    fix["season"] = "20" + s[0:2] + "-" + s[2:4]
    fix.columns = [
        "Match Number",
        "Round Number",
        "Date",
        "Location",
        "HomeTeam",
        "AwayTeam",
        "Result",
        "season",
    ]
    fix = replace_team_names(fix, "HomeTeam")
    fix = replace_team_names(fix, "AwayTeam")
    return fix


def extract_current_fixtures(transformed_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract upcoming fixtures from transformed data and match them with
    fixture times.

    Parameters:
    - transformed_data (pd.DataFrame): The transformed data containing
    historical match details.

    Returns:
    - Optional[pd.DataFrame]: A DataFrame containing the upcoming fixtures
    with additional data, or None if no fixtures are found.

    Notes:
    - Retrieves fixture times using the `get_fixture_times` function.
    - Merges the fixtures with transformed data on season, home team, and
    away team.
    - Asserts that the merged DataFrame contains data; raises AssertionError
    otherwise.
    - Processes fixture dates and times.
    - Filters fixtures based on the current date and returns the next 12
    fixtures.
    """
    fix = get_fixture_times()
    if fix is not None:
        fix = fix.drop(["Result"], axis=1)
        transformed_data["season"] = transformed_data["season"].replace("season_", "")
        current_fixtures = pd.merge(
            transformed_data, fix, on=["season", "HomeTeam", "AwayTeam"], how="inner"
        )
        assert current_fixtures.shape[0] > 0, "No matching fixtures found"
        current_fixtures["Fixture Date"] = pd.to_datetime(
            current_fixtures["Date"].str.slice(0, 10), format="%d/%m/%Y"
        )
        current_fixtures["Fixture Time"] = (
            current_fixtures["Date"].str.slice(11, 13).astype(int) + 1
        ).astype(str) + current_fixtures["Date"].str.slice(13, 16)
        c_date = pd.to_datetime(
            datetime.datetime.now().strftime("%Y-%m-%d"), format="%Y-%m-%d"
        )
        current_fixtures = current_fixtures.sort_values(
            by=["Fixture Date", "Fixture Time"]
        )
        current_fixtures = current_fixtures[current_fixtures["Fixture Date"] >= c_date]
        current_fixtures["Fixture"] = (
            current_fixtures["HomeTeam"] + " v " + current_fixtures["AwayTeam"]
        )
        current_fixtures = current_fixtures.head(12)
        return current_fixtures
    else:
        return None


def get_fixtures(full_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Append upcoming fixture data to the full dataset of Premier League matches.

    Parameters:
    - full_data (pd.DataFrame): The existing dataset containing historical
    match data.

    Returns:
    - Optional[pd.DataFrame]: The updated dataset including upcoming fixtures,
    or None if no fixtures are found.

    Notes:
    - Uses the `extract_season` function to determine the current season.
    - Fetches fixture data for the current season and appends it to the existing
    dataset.
    - Fills missing goal data with NaNs and the match result with
    'D' (for draw) as placeholders.
    - Standardizes team names using the `replace_team_names` function.
    - Removes duplicate entries based on season, date, home team, and away team.
    """
    s = extract_season()
    s1 = s[0:2]
    csv_name = f"https://fixturedownload.com/download/epl-20{s1}-UTC.csv"
    logger_config.logger.info(f"Loading data from {csv_name}")
    try:
        fix = pd.read_csv(csv_name)
        logger_config.logger.info(f"Loaded data from {csv_name}")
        fix = fix[fix["Result"].isna()]
        fix["season"] = "20" + s[0:2] + "-" + s[2:4]
        fix = fix[["season", "Date", "Home Team", "Away Team"]]
        fix.columns = ["season", "Date", "HomeTeam", "AwayTeam"]
        fix["Date"] = fix["Date"].str.slice(0, 10)
        fix["FTHG"] = np.nan
        fix["FTAG"] = np.nan
        fix["FTR"] = np.nan
        full_data = pd.concat([full_data, fix])
        full_data = replace_team_names(full_data, "HomeTeam")
        full_data = replace_team_names(full_data, "AwayTeam")
        full_data["FTHG"] = full_data["FTHG"].fillna(0)
        full_data["FTAG"] = full_data["FTAG"].fillna(0)
        full_data["FTR"] = full_data["FTR"].fillna("D")
        full_data = full_data.drop_duplicates(
            subset=["season", "Date", "HomeTeam", "AwayTeam"], keep="first"
        )
        return full_data
    except Exception as e:
        logger_config.logger.error(f"Error loading fixtures: {e}")
        return None
