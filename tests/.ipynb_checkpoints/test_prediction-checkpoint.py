import pandas as pd
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
try:
    from premier_league import prediction
except ModuleNotFoundError:
    import prediction

class MockRegressionModel:
    def __init__(self):
        self.feature_names_ = ['feature1', 'feature2']

    def predict(self, data):
        return np.array([2.5, 1.8, 0.4])

def test_predict_regression_model(mocker):
    # Create a mock regression model
    mock_model = MockRegressionModel()

    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

    # Call the function
    predictions = prediction.predict(data, mock_model)

    # Assert the predictions
    expected_predictions = np.array([3, 2, 0])
    assert predictions.all() == expected_predictions.all()

def test_predict_classification_model(mocker):
    # Mock the model for a classification case
    mock_model = Mock()
    mock_model.feature_names_ = ['feature1', 'feature2']
    mock_model.predict.return_value = [[0.1, 0.9], [0.6, 0.4], [0.2, 0.8]]

    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

    # Call the function
    predictions = prediction.predict(data, mock_model)

    # Assert the predictions
    expected_predictions = [0.1, 0.9, 0.6, 0.4, 0.2, 0.8]
    assert predictions == expected_predictions

def test_add_res_prediction_home_win(mocker):
    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Test case for home win
    input_data = {'Home Prediction': 2, 'Away Prediction': 1}
    result = prediction.add_res_prediction(input_data)
    assert result == "H"

def test_add_res_prediction_away_win(mocker):
    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Test case for away win
    input_data = {'Home Prediction': 1, 'Away Prediction': 2}
    result = prediction.add_res_prediction(input_data)
    assert result == "A"

def test_add_res_prediction_draw(mocker):
    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Test case for draw
    input_data = {'Home Prediction': 1, 'Away Prediction': 1}
    result = prediction.add_res_prediction(input_data)
    assert result == "D"

