from datetime import datetime

# Current time
current_time = datetime.now().strftime("%Y%m%d")

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
TRANSFORMED_DATA_LOCATION = "app_data/transformed_data_cc.csv"
HISTORICAL_DATA_DRIFT =  "app_data/historical_pl_data.csv"
MODEL_PERFORMANCE_LOCATION = "app_data/model_performance/model_performance_data.csv"

# Logging
LOG_FOLDER = 'app_data/logging'
LOG_LEVEL = 'INFO'

# Data expectations
RUN_DATA_EXPECTATIONS = True
EXPECTATIONS_LOCATION = "../data/expectations"
EXP_LOC = '../data/expectations/exp_prem_results.json'
VALIDATION_TOPIC = 'arn:aws:sns:eu-west-2:372535189767:sns-data-validation-alert'
VALIDATION_RESULTS_PATH = f'app_data/expectations/valid_results_{current_time}.json'

# Data drift
DRIFT_REPORT_LOC = f"app_data/data_drift_reports/drift_report_{current_time}.html"
DRIFT_TOPIC = 'arn:aws:sns:eu-west-2:372535189767:sns-data-drift-report'

# Modelling
MODEL_VERSION = 'v2'
TRANSFORMER_PATH = f'app_data/transformers/transformer_{MODEL_VERSION}_{current_time}.pkl'
CLASS_MODEL_NAME = f'app_data/models/classifier_{MODEL_VERSION}_{current_time}.pkl'
HOME_MODEL_NAME = f'app_data/models/home_regress_model_{MODEL_VERSION}_{current_time}.pkl'
AWAY_MODEL_NAME = f'app_data/models/away_regress_model_{MODEL_VERSION}_{current_time}.pkl'
MAX_EVALS = 20

# Names
TEAM_NAME_REPLACEMENTS = {'Nottingham Forest': "Nott'm Forest",
                          'Spurs': "Tottenham",
                          'Man Utd': "Man United",
                          'Sheffield Utd': 'Sheffield United'}

# App
BADGE_SCALE_FACTOR = 0.5

# Postgres
POSTGRES_DB_ID = 'premier-league-logging'

# MLFlow
EXP_NAME = "premier-league-experiments"
