from typing import Optional
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from PIL import Image
import io
import os
from io import (BytesIO, StringIO)
import pickle
from datetime import datetime

# import constants
from premier_league import constants as constants


def get_current_date_time() -> str:
    """
    Returns the current local date and time as a string.
    
    :return: A string representing the current date and time in the format YYYY-MM-DD HH:MM:SS
    """
    # Get the current local date and time
    now = datetime.now()
    
    # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    return date_time_str


def grab_data_s3(
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app",
) -> pd.DataFrame:
    """
    Retrieve data from an S3 bucket and return a DataFrame.

    :param bucket: Name of the S3 bucket.
    :type bucket: str
    :param file_path: Path to the file within the S3 bucket.
    :type file_path: str
    :param profile_name: AWS profile name (optional).
    :type profile_name: Optional[str]
    :return: DataFrame containing the data from the S3 file.
    :rtype: pd.DataFrame
    :raises NoCredentialsError: If credentials are missing.
    :raises PartialCredentialsError: If credentials are incomplete.
    """
    try:
        if constants.LOCAL_MODE:
            session = boto3.Session(profile_name="premier-league-app")
        else:
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2",
            )
        s3 = session.client("s3")  # Create a connection to S3
        obj = s3.get_object(
            Bucket=bucket, Key=file_path
        )  # Get object and file from bucket
        data = pd.read_csv(obj["Body"])
        return data
    except NoCredentialsError:
        raise NoCredentialsError(
            """Credentials not available. Make sure the profile
            name is correct and the credentials are set up properly."""
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def save_data_s3(
    data: pd.DataFrame,
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    file_format: str = "csv",
    profile_name: Optional[str] = "premier-league-app"
) -> None:
    """
    Save a DataFrame to an S3 bucket.

    :param data: DataFrame to save.
    :type data: pd.DataFrame
    :param bucket: Name of the S3 bucket.
    :type bucket: str
    :param file_path: Path to the file within the S3 bucket.
    :type file_path: str
    :param file_format: Format of the file to save ('csv', 'excel', etc.).
    :type file_format: str
    :param profile_name: AWS profile name (optional).
    :type profile_name: Optional[str]
    :raises NoCredentialsError: If credentials are missing.
    :raises PartialCredentialsError: If credentials are incomplete.
    """
    try:
        if profile_name:
            # Use the specified profile
            session = boto3.Session(profile_name=profile_name)
        else:
            # Use environment variables or default profile
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION")
            )
        s3_resource = session.resource("s3")
        if file_format.lower() == 'csv':
            # Save DataFrame as a CSV
            buffer = StringIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)
            s3_resource.Bucket(bucket).put_object(Key=file_path, Body=buffer.getvalue())
        elif file_format.lower() == 'excel':
            # Save DataFrame as an Excel file
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False)
            buffer.seek(0)
            s3_resource.Bucket(bucket).put_object(Key=file_path, Body=buffer.getvalue())
        else:
            raise ValueError(f"File format '{file_format}' is not supported.")
    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def load_transformer_s3_pickle(
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app"
):
    """
    Load a scikit-learn transformer object from a pickle file on an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param file_path: Path where the pickle file is stored within the S3 bucket.
    :param profile_name: AWS profile name (optional).
    :return: The loaded transformer object.
    """
    try:
        # The profile name is optional and used for local development
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            # Use environment variables for AWS credentials
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2"  # or your AWS region
            )
        
        s3 = session.client("s3")
        response = s3.get_object(Bucket=bucket, Key=file_path)
        
        # Read the object (which is file-like) using BytesIO
        with BytesIO(response['Body'].read()) as f:
            # Load the transformer object from the BytesIO object using pickle
            transformer = pickle.load(f)
        
        return transformer

    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def save_transformer_s3_pickle(
    transformer,
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app"
) -> None:
    """
    Save a scikit-learn transformer object to a pickle file on an S3 bucket.

    :param transformer: Transformer object to save.
    :param bucket: Name of the S3 bucket.
    :param file_path: Path where the pickle file will be saved within the S3 bucket.
    :param profile_name: AWS profile name (optional).
    """
    try:
        # The profile name is optional and used for local development
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            # Use environment variables for AWS credentials
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2"  # or your AWS region
            )
        
        s3_client = session.client("s3")
        with BytesIO() as f:
            # Save transformer object to a BytesIO object using pickle
            pickle.dump(transformer, f)
            f.seek(0)  # Move pointer to the start of the file
            s3_client.put_object(
                Bucket=bucket, Key=file_path, Body=f.getvalue())
            print(f"Transformer object is saved to S3 bucket {bucket} at {file_path}")

    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )