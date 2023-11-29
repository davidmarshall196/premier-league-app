import pytest
from botocore.exceptions import ClientError
from unittest.mock import patch, MagicMock
try:
    from premier_league import postgres
except ModuleNotFoundError:
    import postgres

def test_get_instance_status(mocker):
    # Mocking constants.LOCAL_MODE
    mocker.patch('premier_league.constants.LOCAL_MODE', new=True)

    # Mocking boto3 session and client
    mock_session = mocker.patch('boto3.Session')
    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client

    # Mocking client's describe_db_instances method
    mock_response = {
        'DBInstances': [
            {'DBInstanceStatus': 'available'}
        ]
    }
    mock_client.describe_db_instances.return_value = mock_response

    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Call the function
    status = postgres.get_instance_status('test-instance')

    # Assert the status
    assert status == 'available'
    mock_client.describe_db_instances.assert_called_once_with(DBInstanceIdentifier='test-instance')

def test_get_instance_status_no_instance(mocker):
    # Mocking constants.LOCAL_MODE
    mocker.patch('premier_league.constants.LOCAL_MODE', new=True)

    # Mocking boto3 session and client
    mock_session = mocker.patch('boto3.Session')
    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client

    # Mocking client's describe_db_instances method
    mock_response = {
        'DBInstances': [
            {'DBInstanceStatus': 'available'}
        ]
    }
    # Setup mock response with no instances
    mock_client.describe_db_instances.return_value = {'DBInstances': []}

    # Call the function and expect an exception
    with pytest.raises(Exception) as excinfo:
        postgres.get_instance_status('test-instance')

    assert "No RDS instance found" in str(excinfo.value)

def test_get_instance_status_client_error(mocker):
    # Mocking constants.LOCAL_MODE
    mocker.patch('premier_league.constants.LOCAL_MODE', new=True)

    # Mocking boto3 session and client
    mock_session = mocker.patch('boto3.Session')
    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client

    # Mocking client's describe_db_instances method
    mock_response = {
        'DBInstances': [
            {'DBInstanceStatus': 'available'}
        ]
    }
    # Setup mock client to raise ClientError
    mock_client.describe_db_instances.side_effect = ClientError(
        {'Error': {'Code': 'SomeError', 'Message': 'Error message'}},
        'DescribeDBInstances'
    )

    # Call the function and expect an exception
    with pytest.raises(Exception) as excinfo:
        postgres.get_instance_status('test-instance')

    assert "Error retrieving RDS instance status" in str(excinfo.value)

def test_start_rds_instance(mocker):
    # Mocking constants.LOCAL_MODE
    mocker.patch('premier_league.constants.LOCAL_MODE', new=True)

    # Mocking boto3 session and client
    mock_session = mocker.patch('boto3.Session')
    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client

    # Mocking client's start_db_instance method
    mock_client.start_db_instance.return_value = None

    # Mocking wait_for_instance_status function
    mocker.patch('premier_league.postgres.wait_for_instance_status')

    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Call the function
    postgres.start_rds_instance('test-instance')

    # Assert that start_db_instance was called
    mock_client.start_db_instance.assert_called_once_with(
        DBInstanceIdentifier='test-instance')
    # Assert that wait_for_instance_status was called
    postgres.wait_for_instance_status.assert_called_once_with(
        'test-instance', 'available')

def test_start_rds_instance_client_error(mocker):
    # Mocking constants.LOCAL_MODE
    mocker.patch('premier_league.constants.LOCAL_MODE', new=True)

    # Mocking boto3 session and client
    mock_session = mocker.patch('boto3.Session')
    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client

    # Setup mock client to raise ClientError
    mock_client.start_db_instance.side_effect = ClientError(
        {'Error': {'Code': 'SomeError', 'Message': 'Error message'}},
        'StartDBInstance'
    )

    # Mocking logger
    mock_logger = mocker.patch('premier_league.logger_config.logger')

    # Call the function and expect an exception
    with pytest.raises(ClientError):
        postgres.start_rds_instance('test-instance')

    # Assert that an error log was created
    mock_logger.error.assert_called()