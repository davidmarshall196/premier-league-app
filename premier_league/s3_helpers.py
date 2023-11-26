from typing import Optional
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.client import Config
from PIL import Image
import io
import os
from io import BytesIO, StringIO
import pickle
from datetime import datetime
import streamlit as st
import json

# import constants
try:
    from premier_league import constants, logger_config
except ImportError:
    import constants
    import logger_config


def get_current_date_time() -> str:
    """
    Returns the current local date and time as a string.

    Parameters:
    - None

    Returns:
    - str: A string representing the current date and time in the format
    YYYY-MM-DD HH:MM:SS
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

    Parameters:
    - file_path (str): Path to the file within the S3 bucket.
    - bucket (str): Name of the S3 bucket. Defaults to the value in constants.S3_BUCKET.
    - profile_name (Optional[str]): AWS profile name. Defaults to "premier-league-app".

    Returns:
    - pd.DataFrame: DataFrame containing the data from the S3 file.

    Raises:
    - NoCredentialsError: If AWS credentials are missing.
    - PartialCredentialsError: If AWS credentials are incomplete.
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
        logger_config.logger.info(f"Loading data from {bucket}/{file_path}")
        obj = s3.get_object(
            Bucket=bucket, Key=file_path
        )  # Get object and file from bucket
        data = pd.read_csv(obj["Body"])
        logger_config.logger.info(f"Successfully loaded data from {bucket}/{file_path}")
        return data
    except NoCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred reading {bucket}/{file_path}: %s", str(e)
        )
        raise NoCredentialsError(
            """Credentials not available. Make sure the profile
            name is correct and the credentials are set up properly."""
        )
    except PartialCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred reading {bucket}/{file_path}: %s", str(e)
        )
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        logger_config.logger.error(
            f"An error occurred reading {bucket}/{file_path}: %s", str(e)
        )
        raise Exception(f"An unexpected error occurred: {str(e)}")


def save_data_s3(
    data: pd.DataFrame,
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    file_format: str = "csv",
    profile_name: Optional[str] = "premier-league-app",
) -> None:
    """
    Save a DataFrame to an S3 bucket.

    Parameters:
    - data (pd.DataFrame): DataFrame to save.
    - file_path (str): Path to the file within the S3 bucket.
    - bucket (str): Name of the S3 bucket. Defaults to the value in
    constants.S3_BUCKET.
    - file_format (str): Format of the file to save ('csv', 'excel', etc.). Defaults
    to "csv".
    - profile_name (Optional[str]): AWS profile name. Defaults to "premier-league-app".

    Raises:
    - NoCredentialsError: If AWS credentials are missing.
    - PartialCredentialsError: If AWS credentials are incomplete.
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
                region_name=os.getenv("AWS_DEFAULT_REGION"),
            )
        s3_resource = session.resource("s3")
        logger_config.logger.info(f"Savng data to {bucket}/{file_path}")
        if file_format.lower() == "csv":
            # Save DataFrame as a CSV
            buffer = StringIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)
            s3_resource.Bucket(bucket).put_object(Key=file_path, Body=buffer.getvalue())
        elif file_format.lower() == "excel":
            # Save DataFrame as an Excel file
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                data.to_excel(writer, index=False)
            buffer.seek(0)
            s3_resource.Bucket(bucket).put_object(Key=file_path, Body=buffer.getvalue())
        else:
            raise ValueError(f"File format '{file_format}' is not supported.")
    except NoCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred saving {bucket}/{file_path}: %s", str(e)
        )
        raise NoCredentialsError(
            """Credentials not available. Make sure the profile name is
            correct and the credentials are set up properly."""
        )
    except PartialCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred saving {bucket}/{file_path}: %s", str(e)
        )
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        logger_config.logger.error(
            f"An error occurred saving {bucket}/{file_path}: %s", str(e)
        )
        raise Exception(f"An unexpected error occurred: {str(e)}")


def save_transformer_s3_pickle(
    transformer,
    file_path: str,
    bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app",
    is_transformer: Optional[bool] = True,
) -> None:
    """
    Save a scikit-learn transformer object to a pickle file on an S3 bucket.

    Parameters:
    - transformer: The transformer object to save.
    - file_path (str): Path where the pickle file will be saved within the
    S3 bucket.
    - bucket (str): Name of the S3 bucket. Defaults to the value in
    constants.S3_BUCKET.
    - profile_name (Optional[str]): AWS profile name. Defaults to "premier-league-app".
    - is_transformer (Optional[bool]): A flag indicating if the object is a
    transformer. Defaults to True.

    Notes:
    - The function saves the transformer object in a pickle format.
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
                region_name="eu-west-2",  # or your AWS region
            )

        s3_client = session.client("s3")
        logger_config.logger.info(f"Saving transformer to {bucket}/{file_path}")
        if is_transformer:
            # Serialise the transformer object to a byte string
            serialised_transformer = pickle.dumps(transformer)

            # Upload the byte string contents to S3
            s3_client.put_object(
                Bucket=bucket, Key=file_path, Body=serialised_transformer
            )
            logger_config.logger.info(f"Transformer saved to {bucket}/{file_path}")

        else:
            with BytesIO() as f:
                # Save transformer object to a BytesIO object using pickle
                pickle.dump(transformer, f)
                f.seek(0)  # Move pointer to the start of the file
                s3_client.put_object(Bucket=bucket, Key=file_path, Body=f.getvalue())
                logger_config.logger.info(f"Transformer saved to {bucket}/{file_path}")

    except NoCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred saving {bucket}/{file_path}: %s", str(e)
        )
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred saving {bucket}/{file_path}: %s", str(e)
        )
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )


def get_latest_model_file(bucket: str, prefix: str, session: boto3.Session) -> str:
    """
    Retrieve the filename of the latest model file from an S3 bucket based on a
    specified prefix.

    Parameters:
    - bucket (str): The name of the S3 bucket.
    - prefix (str): The prefix (folder path) where the model files are stored in the
    S3 bucket.
    - session (boto3.Session): The boto3 session used to interact with S3.

    Returns:
    - str: The filename of the latest model file. Returns None if no files are found.

    Notes:
    - The function assumes that the files are named in a way that includes a date in
    the format '%Y%m%d'.
    - It compares these dates to find the latest file.
    """
    s3_client = session.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    latest_model = None
    latest_date = None

    for obj in response.get("Contents", []):
        filename = obj["Key"]
        file_date_str = filename.split("_")[-1].split(".")[0]
        file_date = datetime.strptime(file_date_str, "%Y%m%d")

        if not latest_date or file_date > latest_date:
            latest_date = file_date
            latest_model = filename

    return latest_model


def load_transformer_s3_pickle(
    prefix: str,
    bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app",
    is_transformer: Optional[bool] = True,
) -> object:
    """
    Load the latest scikit-learn transformer object from a pickle file on an S3 bucket.

    Parameters:
    - prefix (str): Prefix of the file path where the pickle files are stored in the
    S3 bucket.
    - bucket (str): Name of the S3 bucket. Defaults to the value in
    constants.S3_BUCKET.
    - profile_name (Optional[str]): AWS profile name. Defaults to "premier-league-app".
    - is_transformer (Optional[bool]): Flag to indicate if the object is a transformer.
    Defaults to True.

    Returns:
    - object: The loaded transformer object.

    Notes:
    - Retrieves the latest file based on the provided prefix.
    - Assumes the transformer object is stored in a pickle format.
    - Uses AWS credentials from the provided profile or environment variables.
    """
    try:
        # Session setup
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2",  # or your AWS region
            )

        # Get the latest model file
        logger_config.logger.info(f"Grabbing transformer from {bucket}/{prefix}")
        latest_file = get_latest_model_file(bucket, prefix, session)
        if not latest_file:
            raise FileNotFoundError("No model files found with the given prefix.")

        s3 = session.client("s3")
        response = s3.get_object(Bucket=bucket, Key=latest_file)

        if is_transformer:
            serialised_transformer = response["Body"].read()
            transformer = pickle.loads(serialised_transformer)
        else:
            with BytesIO(response["Body"].read()) as f:
                transformer = pickle.load(f)
        logger_config.logger.info(f"Loaded transformer from {bucket}/{prefix}")

        return transformer

    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Ensure the profile "
            "name is correct and credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def load_and_display_image_from_s3(
    team_name: str,
    bucket_name: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app",
) -> Image:
    """
    Load an image from an S3 bucket and return it.

    Parameters:
    - team_name (str): The name of the team, used to construct the object name in S3.
    - bucket_name (str): Name of the S3 bucket. Defaults to the value in
    constants.S3_BUCKET.
    - profile_name (Optional[str]): AWS profile name for local development.
    Defaults to "premier-league-app".

    Returns:
    - Image: An image object.

    Notes:
    - The function loads an image based on the team name from the specified S3 bucket.
    - AWS credentials are taken from the specified profile or environment variables.
    - The image is returned as a PIL Image object.

    Raises:
    - NoCredentialsError: If AWS credentials are missing.
    - PartialCredentialsError: If AWS credentials are incomplete.
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
                region_name="eu-west-2",  # or your AWS region
            )
        s3_client = session.client("s3")
        # Get the image object from S3
        object_name = team_name.replace("'", "").replace(" ", "_")
        logger_config.logger.info(f"Loading image from {bucket_name}/{object_name}")
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        image_content = response["Body"].read()

        # Open the image and resize it
        image = Image.open(io.BytesIO(image_content))
        logger_config.logger.info(f"Loaded image from {bucket_name}/{object_name}")
        return image

    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )


def display_side_by_side_images(
    fixture: str,
    bucket_name: str = constants.S3_BUCKET,
    scale_factor: float = constants.BADGE_SCALE_FACTOR,
) -> None:
    """
    Display two images side by side for a given fixture.

    Parameters:
    - fixture (str): The fixture for which the team badges will be displayed.
    - bucket_name (str): Name of the S3 bucket. Defaults to the value in
    constants.S3_BUCKET.
    - scale_factor (float): A factor to scale the images. Defaults to
    constants.BADGE_SCALE_FACTOR.

    Notes:
    - The function assumes that team badges are stored in an S3 bucket with a
    specific naming pattern.
    - It loads and resizes the badges of the two teams involved in the fixture and
    displays them side by side.
    """
    image_1 = fixture.split(" v ")[0].replace(" ", "_") + ".png"
    image_2 = fixture.split(" v ")[1].replace(" ", "_") + ".png"
    image_1 = f"app_data/badges/{image_1}"
    image_2 = f"app_data/badges/{image_2}"
    image_names = [image_1, image_2]

    images = [load_and_display_image_from_s3(name, bucket_name) for name in image_names]

    # Find the max height of the two images and apply the scale factor
    max_height = int(max(image.size[1] for image in images) * scale_factor)

    # Resize images to have the scaled height while preserving aspect ratio
    resized_images = [
        image.resize((int(image.width * max_height / image.height), max_height))
        for image in images
    ]

    total_width = sum(image.size[0] for image in resized_images)
    combined_image = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))
    x_offset = 0
    for image in resized_images:
        # Convert to 'RGBA' if necessary
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        combined_image.paste(
            image, (x_offset, 0), image
        )  # Use image itself as mask for transparency
        x_offset += image.size[0]

    # Display the combined image in Streamlit
    st.image(combined_image)


def json_to_s3(data_dict: dict, bucket_name: str, object_key: str) -> None:
    """
    Uploads a Python dictionary as a JSON file to an S3 bucket.

    Parameters:
    - data_dict (dict): The dictionary to upload.
    - bucket_name (str): The name of the S3 bucket.
    - object_key (str): The key (filename) under which to store the object
                        in the S3 bucket.
    Returns:
    - None
    """
    # Initialise S3 client with config
    config = Config(signature_version="s3v4", region_name="eu-west-2")
    s3 = boto3.client("s3", config=config)

    # Convert the dictionary to a JSON string
    logger_config.logger.info(f"Saving JSON to {bucket_name}/{object_key}")
    json_str = json.dumps(data_dict)

    # Upload the JSON string to the specified S3 bucket
    s3.put_object(Body=json_str, Bucket=bucket_name, Key=object_key)
    logger_config.logger.info(f"Saved JSON to {bucket_name}/{object_key}")
