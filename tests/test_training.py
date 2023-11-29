import pytest
from unittest.mock import MagicMock
import catboost as ctb
import mlflow
import pandas as pd
try:
    from premier_league import training
except ModuleNotFoundError:
    import training

def test_add_result_prediction():
    # Creating sample input data
    input_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    # Creating sample predictions
    # This should be in a format compatible with concatenation to input_data
    predictions = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

    # Calling the function
    result = training.add_result_prediction(
        input_data, predictions)

    # Asserting that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Asserting that the result has the expected shape
    expected_rows = input_data.shape[0] + predictions.shape[0]
    expected_columns = max(input_data.shape[1], predictions.shape[1])  
    
    # Assuming concatenation along index (axis 0)
    assert result.shape == (expected_rows, expected_columns)


# Mock CatBoost classes as types
class MockCatBoostClassifier:
    def __init__(self, *args, **kwargs):
        pass

class MockCatBoostRegressor:
    def __init__(self, *args, **kwargs):
        pass

# Test case for invalid model type
def test_train_model_invalid_type(mocker):
    mocker.patch.object(ctb, 'CatBoostClassifier', MockCatBoostClassifier)
    mocker.patch.object(ctb, 'CatBoostRegressor', MockCatBoostRegressor)
    mocker.patch.object(mlflow, 'start_run', autospec=True)
    mocker.patch.object(mlflow, 'log_params', autospec=True)
    mocker.patch.object(mlflow, 'log_param', autospec=True)
    mocker.patch.object(mlflow.catboost, 'log_model', autospec=True)
    mocker.patch('pandas.DataFrame')
    mocker.patch('pandas.Series')
    mocker.patch('premier_league.logger_config.logger.info')
    x_train = pd.DataFrame()
    y_train = pd.Series()
    with pytest.raises(ValueError):
        training.train_model(x_train, y_train, model_type="invalid_type")


