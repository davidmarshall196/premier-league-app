"""Functions related to training models."""
import pickle
from typing import Union
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
from typing import Dict, Any
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import train_test_split
import catboost as ctb
import numpy as np
import mlflow
import mlflow.catboost

# import constants
try:
    from premier_league import constants, logger_config
except ImportError:
    import constants
    import logger_config


def optimise_hyperparameters(
    training_data: pd.DataFrame,
    target_column: str,
    classification: bool = True,
    max_evals: int = 5,
) -> Dict[str, Any]:
    """
    Optimise CatBoost hyperparameters using Hyperopt.

    Parameters:
    - training_data (pd.DataFrame): The training dataset.
    - target_column (str): The name of the target column.
    - classification (bool): Whether it is a classification task (True)
            or regression task (False).
    - max_evals (int): The number of iterations in the optimisation.

    Returns:
    - Dict[str, Any]: Dictionary containing the best hyperparameters.

    """

    X = training_data.drop(target_column, axis=1)
    y = training_data[target_column]
    learning_rate = np.linspace(0.01, 0.1, 10)
    max_depth = np.arange(2, 12, 2)
    iterations = np.arange(100, 501, 100)

    # Define the categorical features
    categorical_features = list(X.select_dtypes("category").columns)

    ctb_clf_params = {
        "learning_rate": hp.choice("learning_rate", learning_rate),
        "max_depth": hp.choice("max_depth", max_depth),
        "iterations": hp.choice("iterations", iterations),
        "loss_function": "MultiClass" if classification else "RMSE",
    }

    ctb_fit_params = {
        "early_stopping_rounds": 3,
        "verbose": False,
        "cat_features": categorical_features,
    }

    ctb_para = {"clf_params": ctb_clf_params, "fit_params": ctb_fit_params}

    class HYPOpt(object):
        def __init__(self, x_train, x_test, y_train, y_test):
            logger_config.logger.info("Optimising Hyperparameters")
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

        def process(self, fn_name, space, trials, algo, max_evals):
            fn = getattr(self, fn_name)
            try:
                logger_config.logger.info("Entering fmin")
                result = fmin(
                    fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials
                )
            except Exception as e:
                return {"status": STATUS_FAIL, "exception": str(e)}
            return result

        def ctb_clf(self, para):
            clf = (
                ctb.CatBoostClassifier(**para["clf_params"])
                if classification
                else ctb.CatBoostRegressor(**para["clf_params"])
            )
            logger_config.logger.info("CatBoost initialized")
            return self.train_clf(clf, para)

        def train_clf(self, clf, para):
            logger_config.logger.info("Fitting model")
            if classification:
                clf.fit(
                    self.x_train,
                    self.y_train,
                    eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                    **para["fit_params"],
                )
                pred = clf.predict(self.x_test)
                loss = -f1_score(self.y_test, pred, average="micro")
            else:
                clf.fit(
                    self.x_train,
                    self.y_train,
                    eval_set=[(self.x_test, self.y_test)],
                    **para["fit_params"],
                )
                pred = clf.predict(self.x_test)
                loss = mean_squared_error(self.y_test, pred)

            logger_config.logger.info(f"Loss: {loss}")
            return {"loss": loss, "status": STATUS_OK}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    obj = HYPOpt(X_train, X_test, y_train, y_test)
    ctb_opt = obj.process(
        fn_name="ctb_clf",
        space=ctb_para,
        trials=Trials(),
        algo=tpe.suggest,
        max_evals=max_evals,
    )
    best_param = {}
    best_param["learning_rate"] = learning_rate[ctb_opt["learning_rate"]]
    best_param["iterations"] = iterations[ctb_opt["iterations"]]
    best_param["max_depth"] = max_depth[ctb_opt["max_depth"]]
    return best_param


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Union[dict, None] = None,
    model_type: str = "result",
    verbose: bool = False,
    log_model_to_mlflow: bool = False
) -> Union[ctb.CatBoostClassifier, ctb.CatBoostRegressor]:
    """Train CatBoost Classifier or Regressor, optionally specify
    hyperparameters.

    Parameters:
    - x_train (pandas.DataFrame): Data upon which to train.
    - y_train (pandas.Series): Target column.
    - classification (bool): Whether the problem is classification or
    regression.
    - verbose (bool): Whether to print the verbose output from catboost.

    Returns:
    - Trained CatBoost model.
    """
    logger_config.logger.info("Training Model")
    if model_type.lower() not in ["result", "home", "away"]:
        raise ValueError('Model type should be "result", "home" or "away".')

    # Detect categorical features
    if "category" in x_train.dtypes.values:
        cat_features = list(x_train.select_dtypes("category").columns)
    else:
        cat_features = []

    # Start an MLFlow run
    with mlflow.start_run() as run:
        if model_type.lower() == "result":
            model_class = ctb.CatBoostClassifier
        else:
            model_class = ctb.CatBoostRegressor

        # Create and fit the model
        model = model_class(cat_features=cat_features, **(hyperparameters or {}))
        model.fit(x_train, y_train, verbose=verbose)

        # Log hyperparameters and other relevant information
        logger_config.logger.info("Logging parameters to MLFlow")
        mlflow.log_params(hyperparameters or {})
        mlflow.log_param("Model type", model_type)
        mlflow.log_param("Run Date", constants.current_time)
        mlflow.log_param("verbose", verbose)

        # Log the model
        if log_model_to_mlflow:
            mlflow.catboost.log_model(model, "model")

        # Save run ID
        run_id = run.info.run_id

    return model, run_id


def add_result_prediction(input_data: pd.DataFrame, predictions: Any) -> pd.DataFrame:
    """
    Concatenate input data with predictions.

    Parameters:
    - input_data (pd.DataFrame): The input data.
    - predictions (Any): The predictions to be added to the input data.

    Returns:
    - pd.DataFrame: A DataFrame containing the combined input data
    and predictions.

    Notes:
    - This function assumes that the predictions can be concatenated
    with the input DataFrame.
    - The predictions are concatenated along the index (axis 0).
    """
    result = pd.concat([input_data, predictions])
    return result


def optimise_and_train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Call training pipeline.

    Parameters:
    - x_train (pandas.DataFrame): Transformed data upon which to train.
    - y_train (pandas.Series): Target column.

    Returns:
    - (xgboost.sklearn.XGBClassifier): Trained XGBClassifier and fitted transformers.

    """
    hyperparameters = optimise_hyperparameters(x_train, y_train)
    classifier = train_model(
        x_train,
        y_train,
        hyperparameters,
    )
    return classifier


def save_model(model_path: str, model: Any, transformers: Any) -> None:
    """
    Save a model and its transformers to the specified location.

    Parameters:
    - model_path (str): The directory path where the model and
    transformers will be saved.
    - model (Any): The machine learning model to be saved.
    - transformers (Any): The transformers to be saved alongside the model.

    Notes:
    - The model is saved in a file named 'model.pkl'.
    - The transformers are saved in a file named 'transformers.pkl'.
    - Uses Python's pickle module for serialization.
    """
    with open(f"{model_path}/model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    with open(f"{model_path}/transformers.pkl", "wb") as pickle_file:
        pickle.dump(transformers, pickle_file)
