"""Functions related to training models."""
import pickle
from typing import Union
import pandas as pd
import xgboost
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, f1_score
from typing import Dict, Any, List
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import train_test_split
import catboost as ctb
import numpy as np
import pandas as pd

def optimise_hyperparameters_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameter_ranges: Union[dict, None] = None,
) -> (dict, object):
    """Find Bayesian optimised hyperparameters for XGBoost.

    Args:
        x_train (pandas.DataFrame): Data upon which to train.
        y_train (pandas.Series): Target column.
        hyperparameter_ranges (dict): Dictionary of tuples containing min and max
            values for each hyperparameter to be tuned. This includes the eval_metric,
            bayes_opt_init_points, and bayes_opt_n_iter. The default is as follows:
            {
                "alpha": (0, 500),
                "learning_rate": (0, 1),
                "max_depth": (3, 10),
                "subsample": (0.5, 0.9),
                "eval_metric": "logloss",
                "bayes_opt_init_points": 5,
                "bayes_opt_n_iter": 25,
            }

    Returns:
        (dict): Optimised XGBoost classifier hyperparameters.

    """
    dtrain = xgboost.DMatrix(
        x_train,
        label=y_train,
        enable_categorical=True,
    )

    # This is to avoid using mutable data structures as a default argument
    if not hyperparameter_ranges:
        hyperparameter_ranges = {
            "alpha": (0, 500),
            "learning_rate": (0.001, 1),
            "max_depth": (3, 10),
            "subsample": (0.5, 0.9),
            "eval_metric": "logloss",
            "bayes_opt_init_points": 5,
            "bayes_opt_n_iter": 30,
        }

    evaluation_metric = hyperparameter_ranges.pop("eval_metric")
    bayes_opt_init_points = hyperparameter_ranges.pop("bayes_opt_init_points")
    bayes_opt_n_iter = hyperparameter_ranges.pop("bayes_opt_n_iter")

    # Bayesian Optimization function for xgboost
    # Specify the parameters you want to tune as keyword arguments
    def bo_tune_xgb(
        max_depth,
        learning_rate,
        alpha,
        subsample,
    ):
        """Bayesian Optimisation function for XGBoost."""
        params = {
            "max_depth": int(max_depth),
            "alpha": alpha,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "eval_metric": evaluation_metric,
            "nthread": -1,
        }

        cv_result = xgboost.cv(
            params, dtrain, num_boost_round=70, nfold=5, early_stopping_rounds=5
        )
        return 1 - cv_result[f"test-{evaluation_metric}-mean"].iloc[-1]

    xgb_bo = BayesianOptimization(
        bo_tune_xgb,
        hyperparameter_ranges,
    )

    xgb_bo.maximize(
        init_points=bayes_opt_init_points, n_iter=bayes_opt_n_iter, acq="ei"
    )

    hyperparameters = xgb_bo.max["params"]
    hyperparameters["nthread"] = -1
    hyperparameters["eval_metric"] = evaluation_metric

    # Converting values from float to int
    hyperparameters["max_depth"] = int(hyperparameters["max_depth"])

    return hyperparameters, xgb_bo.res


def optimise_hyperparameters(
    training_data: pd.DataFrame, 
    target_column: str, 
    classification: bool = True,
    max_evals: int = 5
) -> Dict[str, Any]:
    """
    Optimize CatBoost hyperparameters using Hyperopt.

    Args:
        training_data (pd.DataFrame): The training dataset.
        target_column (str): The name of the target column.
        classification (bool): Whether it is a classification task (True) or regression task (False).
        max_evals (int): The number of iterations in the optimisation.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters.

    """

    X = training_data.drop(target_column, axis=1)
    y = training_data[target_column]
    learning_rate = np.linspace(0.01, 0.1, 10)
    max_depth = np.arange(2, 12, 2)
    colsample_bylevel = np.arange(0.3, 0.8, 0.1)
    iterations = np.arange(100, 501, 100)
    l2_leaf_reg = np.arange(0, 10)
    bagging_temperature = np.arange(0, 100, 10)

    # Define the categorical features if any in the dataset for CatBoost to handle
    categorical_features = list(X.select_dtypes('category').columns)

    ctb_clf_params = {
        'learning_rate': hp.choice('learning_rate', learning_rate),
        'max_depth': hp.choice('max_depth', max_depth),
        'iterations': hp.choice('iterations', iterations),
        'loss_function': 'MultiClass' if classification else 'RMSE',
    }

    ctb_fit_params = {
        'early_stopping_rounds': 3,
        'verbose': False,
        'cat_features': categorical_features
    }

    ctb_para = {'clf_params': ctb_clf_params, 'fit_params': ctb_fit_params}

    class HYPOpt(object):
        def __init__(self, x_train, x_test, y_train, y_test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

        def process(self, fn_name, space, trials, algo, max_evals):
            fn = getattr(self, fn_name)
            try:
                print('Entering fmin')
                result = fmin(fn=fn, space=space, 
                              algo=algo, max_evals=max_evals, trials=trials)
            except Exception as e:
                return {'status': STATUS_FAIL, 'exception': str(e)}
            return result

        def ctb_clf(self, para):
            clf = ctb.CatBoostClassifier(
                **para['clf_params']) if classification else ctb.CatBoostRegressor(
                **para['clf_params'])
            print('CatBoost initialized')
            return self.train_clf(clf, para)

        def train_clf(self, clf, para):
            print('Fitting model')
            if classification:
                clf.fit(self.x_train, self.y_train,
                        eval_set=[(self.x_train, 
                                   self.y_train), (self.x_test, self.y_test)],
                        **para['fit_params'])
                pred = clf.predict(self.x_test)
                loss = -f1_score(self.y_test, pred, average='micro')
            else:
                clf.fit(self.x_train, self.y_train, eval_set=[(
                    self.x_test, self.y_test)], **para['fit_params'])
                pred = clf.predict(self.x_test)
                loss = mean_squared_error(self.y_test, pred)

            print(f'Loss: {loss}')
            return {'loss': loss, 'status': STATUS_OK}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    obj = HYPOpt(X_train, X_test, y_train, y_test)
    ctb_opt = obj.process(
        fn_name='ctb_clf', space=ctb_para, trials=Trials(), 
        algo=tpe.suggest, max_evals=max_evals)
    best_param={}
    best_param['learning_rate']=learning_rate[ctb_opt['learning_rate']]
    best_param['iterations']=iterations[ctb_opt['iterations']]
    best_param['max_depth']=max_depth[ctb_opt['max_depth']]
    return best_param

    
def train_model_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Union[dict, None] = None,
    classification: bool = True
) -> xgboost.XGBClassifier:
    """Train XGBoost Classifier, optionally specifiy hyperparemeters, or save model.

    Args:
        x_train (pandas.DataFrame): Data upon which to train.
        y_train (pandas.Series): Target column.
        hyperparameters (dict): Opitionally specify hypeparemeters for model
            (Default value: None)
        save_path (str): Optionally specify full or relative path. If not specified,
            don't save (Default value: None)

    Returns:
        (xgboost.Classifier): Trained XGBoost model.

    """
    if classification:
        if hyperparameters:
            model = xgboost.XGBClassifier(enable_categorical=True,
                                           tree_method='hist'
                                           **hyperparameters)
        else:
            model = xgboost.XGBClassifier(tree_method='hist',
                                           enable_categorical=True)
        model.fit(
        x_train,
        y_train,
        )
    else:
        if hyperparameters:
            model = xgboost.XGBRegressor(enable_categorical=True,
                                           tree_method='hist'
                                           **hyperparameters)
        else:
            model = xgboost.XGBRegressor(tree_method='hist',
                                           enable_categorical=True)
        model.fit(
        x_train,
        y_train,
        )
    return model

def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Union[dict, None] = None,
    classification: bool = True,
    verbose: bool = False
) -> ctb.CatBoostClassifier:
    """Train XGBoost Classifier, optionally specifiy hyperparemeters, or save model.

    Args:
        x_train (pandas.DataFrame): Data upon which to train.
        y_train (pandas.Series): Target column.
        classification: whether the problem is classification or regression.
        verbose: whether to print the verbose output from catboost.

    Returns:
        (xgboost.Classifier): Trained CatBoost model.

    """
    if 'category' in x_train.dtypes.values:
        cat_features = list(x_train.select_dtypes('category').columns)
    else:
        cat_features = []
    
    if classification:
        if hyperparameters:
            model = ctb.CatBoostClassifier(
                cat_features=cat_features,
                **hyperparameters
            )
        else:
            model = ctb.CatBoostClassifier(cat_features=cat_features)
        
        model.fit(x_train, y_train, verbose=verbose)
    else:
        if hyperparameters:
            model = ctb.CatBoostRegressor(
                cat_features=cat_features,
                **hyperparameters
            )
        else:
            model = ctb.CatBoostRegressor(cat_features=cat_features)
        
        model.fit(x_train, y_train, verbose=verbose)
    
    return model

def add_result_prediction(input_data, predictions):
    result = pd.concat([input_data, predictions])
    return result
    

def optimise_and_train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Call training pipeline.

    Args:
        x_train (pandas.DataFrame): Transformed data upon which to train.
        y_train (pandas.Series): Target column.

    Returns:
        (xgboost.sklearn.XGBClassifier): Trained XGBClassifier and fitted transformers.

    """
    hyperparameters = optimise_hyperparameters(x_train, y_train)
    classifier = train_model(
        x_train,
        y_train,
        hyperparameters,
    )
    return classifier


def save_model(model_path: str, model, transformers):
    """Save model and transformers to specified location."""
    with open(f"{model_path}/model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    with open(f"{model_path}/transformers.pkl", "wb") as pickle_file:
        pickle.dump(transformers, pickle_file)
