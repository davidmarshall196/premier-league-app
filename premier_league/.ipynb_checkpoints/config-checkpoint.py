import boto3
import os

# import constants
try:
    from premier_league import constants as constants
except ModuleNotFoundError:
    import constants


if constants.LOCAL_MODE:
    session = boto3.Session(profile_name="premier-league-app")
else:
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-west-2",
    )


def get_parameter(parameter_name):
    ssm_client = session.client("ssm")
    response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
    return response["Parameter"]["Value"]


print("Grabbing Passwords")
RDS_DB_ID = get_parameter("RDS_IDENTIFIER")
RDS_DB_PASSWORD = get_parameter("RDS_PASSWORD")
RDS_ENDPOINT = get_parameter("RDS_ENDPOINT")
