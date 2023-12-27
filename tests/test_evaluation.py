import pytest
from unittest.mock import patch, Mock
import sklearn.metrics as metrics
import mlflow
try:
    from premier_league import evaluation
except ModuleNotFoundError:
    import evaluation

@patch('premier_league.evaluation.mlflow.start_run')
@patch('premier_league.evaluation.metrics')
@patch('premier_league.evaluation.logger_config.logger')
def test_evaluate_model_result(mock_logger, mock_metrics, mock_mlflow, mocker):
    """ Test evaluate_model function for 'result' model_type """
    # Setup mock metrics
    mock_metrics.matthews_corrcoef.return_value = 0.75
    mock_metrics.accuracy_score.return_value = 0.8
    mock_metrics.f1_score.return_value = 0.85

    # Call the function
    predictions = [1, 0, 1, 1, 0]
    actual_values = [1, 1, 1, 0, 0]
    evaluation_metrics = evaluation.evaluate_model(
        predictions, actual_values, model_type="result")

    # Assert the metrics were calculated as expected
    assert evaluation_metrics["mcc"] == 0.75
    assert evaluation_metrics["accuracy"] == 0.8
    assert evaluation_metrics["f1"] == 0.85

@patch('premier_league.evaluation.mlflow.start_run')
@patch('premier_league.evaluation.metrics')
@patch('premier_league.evaluation.logger_config.logger')
def test_evaluate_model_home_away(mock_logger, mock_metrics, mock_mlflow, mocker):
    """ Test evaluate_model function for 'home' or 'away' model_type """
    # Setup mock metrics
    mock_metrics.r2_score.return_value = 0.7
    mock_metrics.median_absolute_error.return_value = 2
    mock_metrics.mean_absolute_error.return_value = 2.5

    # Call the function
    predictions = [3, 2, 4, 5, 3]
    actual_values = [4, 2, 4, 4, 3]
    evaluation_metrics = evaluation.evaluate_model(
        predictions, actual_values, model_type="home")

    # Assert the metrics were calculated as expected
    assert evaluation_metrics["r2_score"] == 0.7
    assert evaluation_metrics["median_ae"] == 2
    assert evaluation_metrics["mean_ae"] == 2.5

def test_evaluate_model_invalid_type():
    """ Test evaluate_model with an invalid model_type """
    with pytest.raises(ValueError):
        evaluation.evaluate_model(
            [], [], model_type="invalid")
