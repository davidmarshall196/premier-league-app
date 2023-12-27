import logging
import os
from datetime import datetime
import boto3

# import constants
try:
    from premier_league import constants as constants
except ImportError:
    import constants

# Define the name for the log file
log_file_name = datetime.now().strftime("premier_league_%Y_%m_%d.log")

# Create a logger
logger = logging.getLogger("PremierLeagueLogger")
logger.setLevel(logging.INFO)  # Set to your preferred logging level

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and set it for the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)


# Custom S3 logging handler (skeleton example)
class S3LoggingHandler(logging.Handler):
    def __init__(self, bucket, s3_path, session):
        logging.Handler.__init__(self)
        self.bucket = bucket
        self.s3_path = s3_path
        self.session = session

    def emit(self, record):
        log_entry = self.format(record)
        s3_client = self.session.client("s3")
        try:
            # Check if the log file already exists in S3
            try:
                # Try to get the current log file
                current_log = s3_client.get_object(
                    Bucket=self.bucket, Key=f"{self.s3_path}/{log_file_name}"
                )
                # If it exists, read the current content
                log_contents = current_log["Body"].read().decode("utf-8")
                # Append new log entry
                log_contents += f"\n{log_entry}"
            except s3_client.exceptions.NoSuchKey:
                # If the log file does not exist, start with the current log entry
                log_contents = log_entry

            # Upload the updated log contents
            s3_client.put_object(
                Bucket=self.bucket,
                Key=f"{self.s3_path}/{log_file_name}",
                Body=log_contents,
            )
        except Exception as e:
            print(f"Failed to upload log to S3: {e}")


# Configure AWS session for S3 logging
if constants.LOCAL_MODE:
    session = boto3.Session(profile_name="premier-league-app")
else:
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-west-2",
    )

# Add S3LoggingHandler to the logger
s3_logging_handler = S3LoggingHandler(
    bucket=constants.S3_BUCKET, s3_path="app_data/logging", session=session
)
s3_logging_handler.setFormatter(formatter)
logger.addHandler(s3_logging_handler)
