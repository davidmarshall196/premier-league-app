import boto3
from botocore.exceptions import ClientError
from typing import Optional
import time

# import constants
try:
    from premier_league import constants, logger_config
except ImportError:
    import constants
    import logger_config


def get_instance_status(
    instance_identifier: str, profile_name: Optional[str] = "premier-league-app"
) -> str:
    """
    Get the current status of an RDS instance.

    Parameters:
    - instance_identifier (str): The identifier of the RDS instance.
    - profile_name (Optional[str]): The name of the profile to use (if not default).

    Returns:
    - str: The current status of the RDS instance.
    """
    try:
        if constants.LOCAL_MODE:
            session = boto3.Session(profile_name=profile_name)
            client = session.client("rds")

        logger_config.logger.info(f"Grabbing instance status of {instance_identifier}")
        response = client.describe_db_instances(
            DBInstanceIdentifier=instance_identifier
        )
        db_instances = response.get("DBInstances", [])

        if not db_instances:
            logger_config.logger.error(
                f"No RDS instance found with identifier {instance_identifier}."
            )
            raise Exception(
                f"No RDS instance found with identifier {instance_identifier}."
            )

        # Assuming there's always only one instance with the given identifier
        instance_info = db_instances[0]
        current_status = instance_info.get("DBInstanceStatus")
        return current_status

    except ClientError as e:
        logger_config.logger.error(f"Error retrieving RDS instance status: {e}")
        raise Exception(f"Error retrieving RDS instance status: {e}")


def wait_for_instance_status(
    instance_identifier: str,
    desired_status: str,
    check_interval: int = 120,
    profile_name: Optional[str] = "premier-league-app",
) -> None:
    """
    Wait for an RDS instance to reach a specified status.

    Parameters:
    - instance_identifier (str): The identifier of the RDS instance.
    - desired_status (str): The status to wait for.
    - check_interval (int): The interval, in seconds, between status checks.
    Defaults to 120 seconds.
    - profile_name (Optional[str]): AWS profile name. Defaults to
    "premier-league-app".

    Notes:
    - Continuously checks the RDS instance's status until it matches the
    desired status.
    - Logs the current and final status of the RDS instance.
    - Uses a Boto3 session for AWS operations.
    - Handles exceptions during status checks.
    """
    session = (
        boto3.Session(profile_name=profile_name)
        if constants.LOCAL_MODE
        else boto3.Session()
    )
    client = session.client("rds")

    while True:
        try:
            response = client.describe_db_instances(
                DBInstanceIdentifier=instance_identifier
            )
            db_instances = response.get("DBInstances", [])
            if len(db_instances) != 1:
                raise Exception("Error finding RDS instance.")

            instance_info = db_instances[0]
            current_status = instance_info.get("DBInstanceStatus")

            logger_config.logger.info(
                f"RDS instance status:'{instance_identifier}': {current_status}"
            )

            if current_status == desired_status:
                logger_config.logger.info(
                    f"RDS instance '{instance_identifier}' is '{desired_status}'."
                )
                break

        except ClientError as e:
            logger_config.logger.error(f"Error checking RDS instance status: {e}")
        time.sleep(check_interval)


def start_rds_instance(
    instance_identifier: str, profile_name: Optional[str] = "premier-league-app"
) -> None:
    """
    Start an RDS instance and wait for it to become available.

    Parameters:
    - instance_identifier (str): The identifier of the RDS instance.
    - profile_name (Optional[str]): AWS profile name. Defaults to
    "premier-league-app".

    Notes:
    - Starts the specified RDS instance.
    - Waits for the instance to reach the 'available' status.
    - Uses a Boto3 session for AWS operations.
    - Logs the process and handles exceptions during the start operation.
    """
    session = (
        boto3.Session(profile_name=profile_name)
        if constants.LOCAL_MODE
        else boto3.Session()
    )
    client = session.client("rds")
    try:
        client.start_db_instance(DBInstanceIdentifier=instance_identifier)
        logger_config.logger.info(f"Starting RDS instance '{instance_identifier}'")
        wait_for_instance_status(instance_identifier, "available")
    except ClientError as e:
        logger_config.logger.error(f"Error starting RDS instance: {e}")


def stop_rds_instance(
    instance_identifier: str, profile_name: Optional[str] = "premier-league-app"
) -> None:
    """
    Stop the specified RDS instance.

    Parameters:
    - instance_identifier (str): The identifier of the RDS instance.
    - profile_name (Optional[str]): AWS profile name. Defaults to
    "premier-league-app".

    Notes:
    - Stops the specified RDS instance.
    - Uses a Boto3 session for AWS operations.
    - Logs the process and handles exceptions during the stop operation.
    """
    session = (
        boto3.Session(profile_name=profile_name)
        if constants.LOCAL_MODE
        else boto3.Session()
    )
    client = session.client("rds")
    try:
        client.stop_db_instance(DBInstanceIdentifier=instance_identifier)
        logger_config.logger.info(f"Stopping RDS instance '{instance_identifier}'")
    except ClientError as e:
        logger_config.logger.error(f"Error stopping RDS instance: {e}")
