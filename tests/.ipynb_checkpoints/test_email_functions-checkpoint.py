import pytest
import os
from unittest.mock import patch, Mock
import boto3
try:
    from premier_league import email_functions
    from premier_league import constants
except ModuleNotFoundError:
    import email_functions
    import constants

def test_data_drift_alert():
    """ Test email body generation for data drift alert """
    alert_type = 'data_drift'
    bucket_name = 'test_bucket'
    s3_object_name = 'test_object'
    url = 'http://example.com/report'
    expected_body = f"""
      Amber Alert
      Data drift Alert
      Details:
      Date: {constants.current_time}

      Please check the project logs for details

      Data drift report saved at s3://{bucket_name}/{s3_object_name}

      Data drift report link (expires in 1 hour): {url}
    """
    assert email_functions.generate_email_body(
        alert_type, bucket_name, s3_object_name, url) == expected_body

def test_data_validation_alert():
    """ Test email body generation for data validation alert """
    alert_type = 'data_validation'
    bucket_name = 'test_bucket'
    s3_object_name = 'test_object'
    url = 'http://example.com/report'
    expected_body = f"""
      Red Alert
      Data validation Alert
      Details:
      Date: {constants.current_time}

      Please check the project logs for details

      Data validation report saved at s3://{bucket_name}/{s3_object_name}

      Data validation report link (expires in 1 hour): {url}
    """
    assert email_functions.generate_email_body(
        alert_type, bucket_name, s3_object_name, url) == expected_body

def test_invalid_alert_type():
    """ Test with invalid alert type """
    with pytest.raises(ValueError):
        email_functions.generate_email_body(
            'invalid_alert', 'test_bucket', 
            'test_object', 'http://example.com/report')

def test_get_s3_client_with_profile(mocker):
    """ Test getting S3 client with a specified profile name """
    mocked_session = mocker.patch('boto3.Session')
    mocked_client = mocker.Mock()
    mocked_session.return_value.client.return_value = mocked_client

    client = email_functions.get_s3_client(
        profile_name="premier-league-app")
    mocked_session.assert_called_with(profile_name="premier-league-app")
    assert client == mocked_client

def test_generate_presigned_s3_url(mocker):
    """ Test generating a presigned S3 URL """
    # Mock the get_s3_client function
    mock_get_s3_client = mocker.patch(
        'premier_league.email_functions.get_s3_client')
    # Mock the S3 client
    mock_s3_client = Mock()
    mock_get_s3_client.return_value = mock_s3_client
    # Mock the generate_presigned_url method
    mock_s3_client.generate_presigned_url.return_value = "http://mockedurl.com"

    url = email_functions.generate_presigned_s3_url(
        bucket_name="test_bucket",
        s3_object_name="test_object",
        expiration=3600,
        profile_name="premier-league-app",
        region="eu-west-2"
    )

    mock_s3_client.generate_presigned_url.assert_called_with(
        "get_object",
        Params={"Bucket": "test_bucket", "Key": "test_object"},
        ExpiresIn=3600,
    )
    assert url == "http://mockedurl.com"

def test_generate_presigned_s3_url_exception(mocker):
    """ Test generating a presigned S3 URL when an exception is raised """
    # Mock the get_s3_client function
    mock_get_s3_client = mocker.patch(
        'premier_league.email_functions.get_s3_client')
    # Mock the S3 client to raise an exception
    mock_get_s3_client.side_effect = Exception("Test Exception")

    url = email_functions.generate_presigned_s3_url(
        bucket_name="test_bucket",
        s3_object_name="test_object",
        expiration=3600,
        profile_name="premier-league-app",
        region="eu-west-2"
    )

    assert url is None



