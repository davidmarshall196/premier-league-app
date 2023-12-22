import constants
import preprocessing
import training
import evaluation
import prediction
import data_extraction
import s3_helpers
import postgres
import mlflow_functions
import expectations_helpers
import email_functions
import data_drift_functions
from mlflow.tracking import MlflowClient

# Load data
df = s3_helpers.grab_data_s3(constants.TRAINING_DATA_LOCATION)

# Add new data
df = data_extraction.add_new_data(
    df, constants.COLUMNS_REQ, constants.TRAINING_DATA_LOCATION
)

# Load latest expectations
data_expectations = expectations_helpers.load_latest_expectations(
    expectations_helpers.latest_exp_file()
)

# Validate
validation_results = expectations_helpers.validate_data(
    df, data_expectations, expectations_path=constants.VALIDATION_RESULTS_PATH
)

# Send email on failure
if not validation_results["success"]:
    email_functions.send_email(
        "data_validation",
        constants.S3_BUCKET,
        expectations_helpers.latest_exp_file(),
        constants.VALIDATION_TOPIC,
    )
    raise Exception("Stopping Pipeline. Validation has failed")

# Load historical data
historical = s3_helpers.grab_data_s3(constants.HISTORICAL_DATA_DRIFT)

# Detect data drift
data_drift = data_drift_functions.DriftDetector(historical, df)

# Report
drift_report = data_drift.check_data_drift(constants.DRIFT_REPORT_LOC)

# Fit transformer
transformers = preprocessing.fit_transformers(df)

# Save transformer
s3_helpers.save_transformer_s3_pickle(transformers, constants.TRANSFORMER_PATH)

# Transform data
transformed_data = preprocessing.transform_data(df, transformers)

# Split data
training_data, testing_data = preprocessing.split_data(transformed_data)

# Optimise
target_column = "FTR"
hyperparameters = training.optimise_hyperparameters(
    training_data, target_column, max_evals=constants.MAX_EVALS
)

# Start postgresDB for model logging
if postgres.get_instance_status(constants.POSTGRES_DB_ID) != "available":
    postgres.start_rds_instance(constants.POSTGRES_DB_ID)

# Open mlflow tracking
mlflow_functions.open_mlflow_tracking(constants.EXP_NAME)

# Train model
classifier, run_id = training.train_model(
    training_data[[col for col in training_data if col != target_column]],
    training_data[target_column],
    hyperparameters=hyperparameters,
)

# Save model
s3_helpers.save_transformer_s3_pickle(
    classifier, constants.CLASS_MODEL_NAME, is_transformer=False
)

# Make predictions
y_test = testing_data[target_column]
x_test = testing_data[[col for col in testing_data if col != target_column]]
predictions = prediction.predict(x_test, classifier)

# Evaluate model
evaluation_metrics = evaluation.evaluate_model(
    predictions, y_test, model_type="result", run_id=run_id
)

# Predict score
transformed_data = prediction.add_match_result(transformed_data, classifier, df)

# Save transformed data
s3_helpers.save_data_s3(transformed_data, constants.TRANSFORMED_DATA_LOCATION)

# Split new data
training_data, testing_data = preprocessing.split_data(transformed_data)

# Optimise home model
hyperparameters = training.optimise_hyperparameters(
    training_data.drop(["FTR", "FTAG"], axis=1),
    "FTHG",
    classification=False,
    max_evals=constants.MAX_EVALS,
)

# Train home model
regressor_1, run_id_home = training.train_model(
    training_data.drop(["FTR", "FTHG", "FTAG"], axis=1),
    training_data["FTHG"],
    model_type="home",
    verbose=False,
    hyperparameters=hyperparameters,
)

# Save model
s3_helpers.save_transformer_s3_pickle(
    regressor_1, constants.HOME_MODEL_NAME, is_transformer=False
)

# Optimise away model
hyperparameters = training.optimise_hyperparameters(
    training_data.drop(["FTR", "FTHG"], axis=1),
    "FTAG",
    classification=False,
    max_evals=constants.MAX_EVALS,
)

# Train away model
regressor_2, run_id_away = training.train_model(
    training_data.drop(["FTR", "FTHG", "FTAG"], axis=1),
    training_data["FTAG"],
    model_type="away",
    verbose=False,
    hyperparameters=hyperparameters,
)

# Save away model
s3_helpers.save_transformer_s3_pickle(
    regressor_2, constants.AWAY_MODEL_NAME, is_transformer=False
)

# Predict home goals
y_test = testing_data["FTHG"]
x_test = testing_data.copy()
predictions_1 = prediction.predict(x_test, regressor_1)

# Evaluate
evaluation_metrics = evaluation.evaluate_model(
    predictions_1, y_test, model_type="home", run_id=run_id_home
)

# Predict away goals
y_test = testing_data["FTAG"]
x_test = testing_data.copy()
predictions_2 = prediction.predict(x_test, regressor_2)

# Evaluate
evaluation_metrics = evaluation.evaluate_model(
    predictions_2, y_test, model_type="away", run_id=run_id_away
)

# Add fixtures
df = data_extraction.get_fixtures(df)

# Transform new data
transformed_data = preprocessing.transform_data(df, transformers)

# Make future predictions for FTR
predictions = prediction.predict(transformed_data, classifier)
transformed_data["match_prediction"] = predictions

# Make home goals predictions
predictions_1 = prediction.predict(transformed_data, regressor_1)

# Make away goals predictions
predictions_2 = prediction.predict(transformed_data, regressor_2)

# Merge together
transformed_data["Home Prediction"] = predictions_1
transformed_data["Away Prediction"] = predictions_2
transformed_data["Result Prediction"] = transformed_data.apply(
    prediction.add_res_prediction, axis=1
)

# Save updated data
s3_helpers.save_data_s3(transformed_data, constants.PREDICTIONS_LOCATION)

# Now save mlflow results to S3
client = MlflowClient()

# Extract runs
runs = mlflow_functions.get_runs_from_experiment(client, experiment_id=1)

# Turn to dataframe
run_df = mlflow_functions.runs_to_dataframe(
    mlflow_functions.get_runs_from_experiment(client, experiment_id=1)
)

# Save model performance
s3_helpers.save_data_s3(run_df, constants.MODEL_PERFORMANCE_LOCATION)

# Close DB instance
postgres.stop_rds_instance(constants.POSTGRES_DB_ID)
