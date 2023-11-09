# Reload all data?
INITIAL_DATA_LOAD = False
LOCAL_MODE = True

# Column names
COLUMNS_REQ = ['season', 'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
PRED_DF_COL_NAMES = ['Home', 'H', 'A', 'Away', 'Date', 'Time', 'Stadium']

# Data locations
S3_BUCKET = 'premier-league-app'
TRAINING_DATA_LOCATION = "app_data/training_data_full.csv"
PREDICTIONS_LOCATION = "app_data/transformed_data_predictions.csv"
STADIUM_DATA_LOCATION = "app_data/stadiums-with-GPS-coordinates.csv"

# Logging
LOG_FOLDER = 'app_data/logging'
LOG_LEVEL = 'INFO'

# Data expectations
RUN_DATA_EXPECTATIONS = False
EXPECTATIONS_LOCATION = "../data/expectations"
EXP_LOC = '../data/expectations/exp_prem_results.json'

# Modelling
MODEL_VERSION = 'v2'
TRANSFORMER_PATH = f'app_data/transformers/transformer_{MODEL_VERSION}.pkl'
CLASS_MODEL_NAME = f'app_data/models/classifier_{MODEL_VERSION}.pkl'
HOME_MODEL_NAME = f'app_data/models/home_regress_model_{MODEL_VERSION}.pkl'
AWAY_MODEL_NAME = f'app_data/models/away_regress_model_{MODEL_VERSION}.pkl'

# Names
TEAM_NAME_REPLACEMENTS = {'Nottingham Forest': "Nott'm Forest",
                          'Spurs': "Tottenham",
                          'Man Utd': "Man United",
                          'Sheffield Utd': 'Sheffield United'}


