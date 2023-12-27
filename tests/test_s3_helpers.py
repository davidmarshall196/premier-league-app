import pytest
import pandas as pd
import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError, 
    PartialCredentialsError
)
try:
    from premier_league import s3_helpers
    from premier_league import constants
    from premier_league import logger_config
except ModuleNotFoundError:
    import s3_helpers
    import constants
    import logger_config
from io import StringIO
import io
from PIL import Image
import os

@pytest.mark.parametrize(
    "input_file_path, input_bucket, mock_csv, expected_output",
    [
        (
            "some/file/path.csv",
            "some-bucket",
            "col1,col2\n1,2\n3,4\n",
            pd.DataFrame({"col1": [1, 3], "col2": [2, 4]}),
        ),
    ],
)
def test_grab_data_s3(mocker, input_file_path, input_bucket, mock_csv, expected_output):
    # Mocking boto3 Session
    mock_session = mocker.patch.object(boto3, "Session")
    mock_s3_client = mock_session.return_value.client.return_value

    # Mocking get_object to return a mock CSV
    mock_s3_client.get_object.return_value = {"Body": StringIO(mock_csv)}

    # Mocking pandas read_csv
    mock_read_csv = mocker.patch("pandas.read_csv", side_effect=pd.read_csv)

    # Execute the function and get the result
    result = s3_helpers.grab_data_s3(
        input_file_path, input_bucket)

    # Validate the result
    pd.testing.assert_frame_equal(result, expected_output)

    # Assert calls were made as expected
    mock_session.assert_called_once()
    mock_s3_client.get_object.assert_called_once_with(
        Bucket=input_bucket, Key=input_file_path
    )
    mock_read_csv.assert_called_once()

