import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
try:
    from premier_league import inference_pipeline
except ModuleNotFoundError:
    import inference_pipeline

@patch('premier_league.constants')
@patch('premier_league.expectations_helpers')
@patch('premier_league.data_extraction')
@patch('joblib.load')
@patch('premier_league.preprocessing')
@patch('premier_league.prediction')
def test_run_inference_pipeline(
    mock_preprocessing, 
    mock_joblib, 
    mock_data_extraction, 
    mock_expectations_helpers, 
    mock_constants,
    mock_prediction
):
    """ Test the run_inference_pipeline function """

    # Setup mock constants
    mock_constants.RUN_DATA_EXPECTATIONS = False
    mock_constants.INITIAL_DATA_LOAD = False
    mock_constants.SAVE_LOCATION = 'mock_save_location.csv'
    mock_constants.COLUMNS_REQ = ['col1', 'col2']
    mock_constants.TRANSFORMER_PATH = 'mock_transformer_path'
    mock_constants.CLASS_MODEL_NAME = 'mock_class_model'
    mock_constants.HOME_MODEL_NAME = 'mock_home_model'
    mock_constants.AWAY_MODEL_NAME = 'mock_away_model'

    # Setup mock data
    mock_full_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_transformed_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    # Mock methods
    mock_data_extraction.load_all_data.return_value = mock_full_data
    mock_data_extraction.add_new_data.return_value = mock_full_data
    mock_data_extraction.get_fixtures.return_value = mock_full_data
    mock_preprocessing.transform_data.return_value = mock_transformed_data
    mock_joblib.load.return_value = MagicMock()
    mock_prediction.predict.side_effect = lambda x, y: ['Pred1', 'Pred2']

    # Call the function
    result = inference_pipeline.run_inference_pipeline()

    # Assertions
    mock_data_extraction.load_all_data.assert_called()
    mock_data_extraction.add_new_data.assert_called()
    mock_data_extraction.get_fixtures.assert_called()
    mock_preprocessing.transform_data.assert_called()
    assert 'match_prediction' in result.columns
    assert 'Home Prediction' in result.columns
    assert 'Away Prediction' in result.columns
    assert 'Result Prediction' in result.columns

