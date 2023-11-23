import pandas as pd
import numpy as np
try:
    from premier_league import (
        constants,
        data_extraction,
        preprocessing,
        prediction,
        visualisations
    )
    if constants.RUN_DATA_EXPECTATIONS:
        from premier_league import expectations_helpers
except ImportError:
    import constants
    import data_extraction
    import preprocessing
    import prediction
    import visualisations
    if constants.RUN_DATA_EXPECTATIONS:
        from expectations_helpers import (
            AutoGreatExpectations,
            view_full_suite,
            view_suite_summary,
            save_expectations,
            load_expectations,
            validate_data
        )
import joblib
from tabulate import tabulate
import runpy
import shap

def run_inference_pipeline():
    """
    The function forms an inference pipeline for a machine learning model. 
    It loads the data, preprocesses it, and predicts the outcomes using pre-trained models.
    
    It doesn't take any input parameters and relies on pre-defined constants for all 
    its configuration needs.
    
    Returns:
        DataFrame: The final processed DataFrame with predictions.
    """
    # Load Expectations
    if constants.run_data_expectations:
        data_expectations = load_expectations(constants.exp_loc)
    
    # Load data
    if constants.initial_data_load:
        full_data = data_extraction.load_all_data(
            constants.seasons_list, 
            constants.save_location,
            constants.columns_req
        )
    else:
        full_data = pd.read_csv(constants.save_location)
    
    # Add the new data
    full_data = data_extraction.add_new_data(
        full_data, 
        constants.columns_req,
        constants.training_data_location
    )
    
    # Extract the fixtures
    full_data = data_extraction.get_fixtures(full_data)
    
    # Validate
    if constants.run_data_expectations:
        validation_results = validate_data(full_data, data_expectations)
    
    # Load preprocessor
    transformers = joblib.load(constants.transformer_path)
    
    # Transform data
    transformed_data = preprocessing.transform_data(
        full_data, transformers
    )
    
    # Make FTR predictions
    classifier = joblib.load(constants.class_model_name)
    predictions = prediction.predict(transformed_data, classifier)
    transformed_data['match_prediction'] = predictions
    
    # Predict Home goals
    regressor_1 = joblib.load(constants.home_model_name)
    predictions_1 = prediction.predict(transformed_data, regressor_1)
    
    # Predict Away goals
    regressor_2 = joblib.load(constants.away_model_name)
    predictions_2 = prediction.predict(transformed_data, regressor_2)
    
    # Add to the data
    transformed_data['Home Prediction'] = predictions_1
    transformed_data['Away Prediction'] = predictions_2
    transformed_data['Result Prediction'] = transformed_data.apply(
        prediction.add_res_prediction, 
        axis = 1
    )
    
    return transformed_data
    
    
    
    
    
    
    
    
    
    







