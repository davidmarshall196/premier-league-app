import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import os
from typing import Optional, Dict, Any

# import constants
try:
    from premier_league import constants, logger_config
except ImportError:
    import constants
    import logger_config


def generate_email_body(
    alert_type: str,
    bucket_name: str,
    s3_object_name: str,
    url: str,
    alert_date: str = constants.current_time,
) -> str:
    """
    Generate the HTML body for an email notification based on the alert type.

    Parameters:
    - alert_type (str): The type of alert, either 'data_drift' or 'data_validation'.
    - bucket_name (str): The name of the S3 bucket.
    - s3_object_name (str): The key of the object within the S3 bucket.
    - url (str): The URL link to the report.
    - alert_date (str): The date of the alert. Defaults to the current time.

    Returns:
    - str: The HTML body for the email.

    Raises:
    - ValueError: If alert_type is not 'data_drift' or 'data_validation'.
    """
    if alert_type not in ("data_drift", "data_validation"):
        raise ValueError("alert_type must me data_drift or data_validation")
    alert = alert_type.replace("_", " ").capitalize()
    return f"""
      {'Red Alert' if alert_type == 'data_validation' else 'Amber Alert'}
      {alert} Alert
      Details:
      Date: {alert_date}

      Please check the project logs for details

      {alert} report saved at s3://{bucket_name}/{s3_object_name}

      {alert} report link (expires in 1 hour): {url}
    """


def get_s3_client(
    profile_name: Optional[str] = "premier-league-app", region: str = "eu-west-2"
) -> boto3.Session.client:
    """
    Retrieve an S3 client instance using a specified AWS
    profile name or environment variables.

    Parameters:
    - profile_name (Optional[str]): The AWS profile name.
            Defaults to 'premier-league-app'.
    - region (str): The AWS region.

    Returns:
    - boto3.Session.client: An S3 client object.
    """
    if constants.LOCAL_MODE:
        session = boto3.Session(profile_name=profile_name)
    else:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
        )
    return session.client("s3")


def generate_presigned_s3_url(
    bucket_name: str,
    s3_object_name: str,
    expiration: int = 3600,
    profile_name: Optional[str] = "premier-league-app",
    region: str = "eu-west-2",
) -> Optional[str]:
    """
    Generate a presigned URL to download a file from an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - s3_object_name (str): The key of the object within the S3 bucket.
    - expiration (int): Time in seconds for the presigned URL to expire.
    - profile_name (Optional[str]): AWS profile name (optional).
    - region (str): AWS region.

    Returns:
    - Optional[str]: A presigned URL to download the file. None if an error occurs.
    """
    try:
        s3 = get_s3_client(profile_name, region)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_object_name},
            ExpiresIn=expiration,
        )
        return url
    except Exception as e:
        logger_config.logger.error(f"Error generating presigned URL: {e}")
        return None


def send_sns_notification(
    Subject: str,
    Message: str,
    Topic: str,
    profile_name: Optional[str] = "premier-league-app",
    region: str = "eu-west-2",
) -> Dict[str, Any]:
    """
    Send a message to a specified AWS SNS topic.

    Parameters:
    - Subject (str): The subject of the message.
    - Message (str): The body of the message.
    - Topic (str): The ARN of the SNS topic.
    - profile_name (Optional[str]): The AWS profile name. Defaults to
        'premier-league-app'.
    - region (str): The AWS region.

    Returns:
    - Dict[str, Any]: The response from the SNS publish action.

    Raises:
    - NoCredentialsError: If AWS credentials are not available or incorrect.
    - PartialCredentialsError: If AWS credentials are incomplete.
    """
    try:
        session = (
            boto3.Session(profile_name=profile_name)
            if profile_name
            else boto3.Session()
        )
        sns = session.client("sns", region)

        logger_config.logger.info(f"Sending SNS message to topic {Topic}")

        response = sns.publish(TopicArn=Topic, Subject=Subject, Message=Message)
        logger_config.logger.info(f"Sent SNS message to topic {Topic}")
        return response

    except NoCredentialsError as e:
        logger_config.logger.error(f"Error sending SNS: {e}")
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError as e:
        logger_config.logger.error(f"Error sending SNS: {e}")
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )
    except Exception as e:
        logger_config.logger.error(f"Error sending SNS: {e}")
        return ["FAIL", e]


def send_email(
    alert_type: str,
    bucket_name: str,
    s3_object_name: str,
    topic: str,
    alert_date: str = constants.current_time,
) -> dict:
    """
    Generate and send an email notification for specific alert types.

    Parameters:
    - alert_type (str): The type of alert, either 'data_drift' or 'data_validation'.
    - bucket_name (str): The name of the S3 bucket.
    - s3_object_name (str): The key of the object within the S3 bucket.
    - topic (str): The target SNS topic.
    - alert_date (str): The date of the alert. Defaults to the current time.

    Returns:
    - dict: Response from the SNS publish action.
    """
    # Generate presigned URL
    url = generate_presigned_s3_url(bucket_name, s3_object_name)

    if url is None:
        raise Exception("Failed to generate presigned URL")

    # Generate email body
    email_body = generate_email_body(
        alert_type, bucket_name, s3_object_name, url, alert_date
    )

    # Define the subject based on alert type
    subject = f"{alert_type.replace('_', ' ').capitalize()} Alert"

    # Send SNS notification
    response = send_sns_notification(Subject=subject, Message=email_body, Topic=topic)

    return response
