import boto3
from botocore.exceptions import ClientError
from typing import Optional

# import constants
try:
    from premier_league import constants as constants
except ImportError:
    import constants

def wait_for_instance_status(
    instance_identifier, 
    desired_status, 
    check_interval=120,
    profile_name: Optional[str] = "premier-league-app"
        
):
    """Wait for the RDS instance to reach a specific status."""
    if constants.LOCAL_MODE:
        session = boto3.Session(
            profile_name=profile_name
        )
    client = session.client('rds')
    while True:
        try:
            response = client.describe_db_instances(
                DBInstanceIdentifier=instance_identifier)
            db_instances = response.get('DBInstances', [])
            if len(db_instances) != 1:
                raise Exception("Error finding RDS instance.")

            instance_info = db_instances[0]
            current_status = instance_info.get('DBInstanceStatus')

            print(f"Current status of RDS instance '{instance_identifier}': {current_status}")

            if current_status == desired_status:
                print(f"RDS instance '{instance_identifier}' reached status '{desired_status}'.")
                break

        except ClientError as e:
            print(f"Error checking RDS instance status: {e}")

        time.sleep(check_interval)  # Wait before the next check

def start_rds_instance(
    instance_identifier,
    profile_name: Optional[str] = "premier-league-app"
):
    """Starts the specified RDS instance and waits for it to become available."""
    if constants.LOCAL_MODE:
        session = boto3.Session(
            profile_name=profile_name
        )
    
    client = session.client('rds')
    try:
        client.start_db_instance(DBInstanceIdentifier=instance_identifier)
        print(f"Starting RDS instance '{instance_identifier}'")
        wait_for_instance_status(instance_identifier, 'available')
    except ClientError as e:
        print(f"Error starting RDS instance: {e}")

def stop_rds_instance(
    instance_identifier,
    profile_name: Optional[str] = "premier-league-app"
):
    """Stops the specified RDS instance."""
    if constants.LOCAL_MODE:
        session = boto3.Session(
            profile_name=profile_name)
    
    client = session.client('rds')
    try:
        client.stop_db_instance(DBInstanceIdentifier=instance_identifier)
        print(f"Stopping RDS instance '{instance_identifier}'")
    except ClientError as e:
        print(f"Error stopping RDS instance: {e}")


