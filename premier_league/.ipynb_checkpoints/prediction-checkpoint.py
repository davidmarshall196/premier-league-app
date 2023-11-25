"""Functions related to predicting."""
import pickle

import pandas as pd
import shap

# import constants
try:
    from premier_league import (
        logger_config
    )
except ImportError:
    import logger_config


def get_top_shap(df: pd.DataFrame, model, encoded_columns: list = None):
    """Get top three shap features that influence positive class predictions.

    Args:
        df (pd.DataFrame): Data upon which to predict.
        model: Model to predict with.
        encoded_columns (list): Columns that have been encoded to multiple columns
            in the data, for example with onehot encoding. Providing this will
            map the many columns back to one in the returned top_shap_values.
            (Default value: None)

    Returns:
        top_shap_values (pd.DataFrame): Contains the top three SHAP values for
            each row as columns.
        shap_explainer (shap.TreeExplainer): Shap Explainer object, for analysis and
            debugging.
        shap_explanation: Shap explanation object (shap_explainer applied to passed
            data). Useful for analysis and debugging.

    """
    shap_explainer = shap.TreeExplainer(model)
    shap_explanation = shap_explainer(df)
    shap_values = pd.DataFrame(
        shap_explanation.values, columns=shap_explanation.feature_names
    )

    if encoded_columns:
        for column in encoded_columns:
            shap_values[column] = shap_values[
                df.columns[df.columns.str.contains(column)]
            ].sum(axis=1)
            shap_values = shap_values.drop(
                df.columns[df.columns.str.contains(column)], axis=1
            )

    top_shap_values = shap_values.apply(
        lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=3
    )
    top_shap_values.columns = [
        "top_shap_value",
        "second_shap_value",
        "third_shap_value",
    ]

    return top_shap_values, shap_explainer, shap_explanation


def predict(data: pd.DataFrame, model) -> list:
    """
    Predicts the target variable using the provided model.

    Args:
        data (pd.DataFrame): The input data for making predictions.
        model: The trained model for prediction.

    Returns:
        list: The list of predicted values.
    """
    logger_config.logger.info("Making Predictions")
    
    data = data[model.feature_names_]
    if 'Regress' in str(type(model)):
        predictions = model.predict(data)
        predictions = predictions.round().astype(int)
    else:
        predictions = model.predict(data)
        predictions = [item for sublist in predictions for item in sublist]
    return predictions


def add_res_prediction(input_data):
    """
    Adds the match result prediction based on home and away predictions.

    Args:
        input_data: A dictionary-like object containing 'Home Prediction' and 'Away Prediction' values.

    Returns:
        str: The predicted match result ('H' for home win, 'A' for away win, 'D' for draw).
    """
    logger_config.logger.info("Adding result prediction")
    if input_data['Home Prediction'] > input_data['Away Prediction']:
        return 'H'
    elif input_data['Home Prediction'] < input_data['Away Prediction']:
        return 'A'
    else:
        return 'D'


def add_match_result(transformed_data, classifier, new_df):
    """
    Adds match prediction, FTHG, and FTAG columns to the transformed data.

    Args:
        transformed_data (pd.DataFrame): The transformed data to add the columns.
        classifier: The classifier model for match prediction.
        new_df (pd.DataFrame): The original data frame containing FTHG and FTAG columns.

    Returns:
        pd.DataFrame: The transformed data with added columns.
    
    Raises:
        Exception: If the length of transformed_data is not equal to the length of new_df.
    """
    if len(transformed_data) == len(new_df):
        transformed_data[
            'match_prediction'] = predict(transformed_data, classifier)
        transformed_data[
            'match_prediction'] = transformed_data[
            'match_prediction'].astype('category')
        transformed_data['FTHG'] = new_df['FTHG']
        transformed_data['FTAG'] = new_df['FTAG']
        return transformed_data
    else:
        raise Exception('Datasets are not aligned. Check.')


def load_model(model_path: str):
    """
    Loads a previously pickled model and transformers.

    Args:
        model_path (str): The path to the directory containing the model and transformers pickle files.

    Returns:
        tuple: A tuple containing the loaded model and transformers.
    """
    with open(f"{model_path}/model.pkl", "rb") as pickle_file:
        model = pickle.load(pickle_file)
    with open(f"{model_path}/transformers.pkl", "rb") as pickle_file:
        transformers = pickle.load(pickle_file)

    return model, transformers
