import pytest
from unittest.mock import patch, Mock, MagicMock
try:
    from premier_league import mlflow_functions
except ModuleNotFoundError:
    import mlflow_functions

def test_open_mlflow_tracking(mocker):
    # Mock the config module
    mocker.patch('premier_league.config.RDS_DB_PASSWORD', new='dummy_password')
    mocker.patch('premier_league.config.RDS_ENDPOINT', new='dummy_endpoint')
    mocker.patch('premier_league.config.RDS_DB_ID', new='dummy_db_id')

    # Mock mlflow module functions
    mock_set_tracking_uri = mocker.patch('mlflow.set_tracking_uri')
    mock_set_experiment = mocker.patch('mlflow.set_experiment')

    # Call the function
    mlflow_functions.open_mlflow_tracking('test_experiment')

    # Build the expected database URI
    expected_db_uri = "postgresql+psycopg2://postgres:dummy_password@dummy_endpoint:5432/dummy_db_id"

    # Assert calls
    mock_set_tracking_uri.assert_called_once_with(expected_db_uri)
    mock_set_experiment.assert_called_once_with('test_experiment')

def test_get_all_experiments():
    # Create a mock for MlflowClient
    mock_client = MagicMock()
    # Mock the search_experiments method
    mock_client.search_experiments.return_value = [
        Mock(name='Experiment1', experiment_id='123'),
        Mock(name='Experiment2', experiment_id='456')
    ]

    # Call the function
    result = mlflow_functions.get_all_experiments(mock_client)

    # Verify that search_experiments was called
    mock_client.search_experiments.assert_called_once()

    # Assert the result
    assert 'Experiment1' in str(result[0]['name'])
    assert result[0]['experiment_id'] == '123'
    assert 'Experiment2' in str(result[1]['name'])
    assert result[1]['experiment_id'] == '456'
