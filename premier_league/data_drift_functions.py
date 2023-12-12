from typing import Dict, Any, Optional
from pandas import DataFrame
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from boto3.exceptions import Boto3Error
from io import StringIO
import os

# import constants
try:
    from premier_league import constants, email_functions, logger_config
except ImportError:
    import constants
    import email_functions
    import logger_config


class DriftDetector:
    """
    Class to detect and report data drift between reference and
    current datasets.

    Attributes:
        reference_data (DataFrame): Reference data to compare.
        current_data (DataFrame): Current data to check against reference data.
    """

    def __init__(self, reference_data: DataFrame, current_data: DataFrame):
        """
        Initialize DriftDetector with reference and current data.

        Args:
            reference_data (DataFrame): Reference dataset.
            current_data (DataFrame): Current dataset.
        """
        self.reference_data = reference_data
        self.current_data = current_data

    def create_report(
        self,
        object_name: str,
        bucket_name: str = constants.S3_BUCKET,
        profile_name: Optional[str] = 'premier-league-app',
    ) -> Dict[str, Any]:
        """
        Create a data drift report and save it directly to S3.

        Args:
            bucket_name (str): Name of the S3 bucket.
            object_name (str): S3 object name (path and filename).
            profile_name (str): AWS profile name (optional).

        Returns:
            Dict[str, Any]: A dictionary containing the data drift
            report.
        """
        try:
            # Setup AWS session
            if constants.LOCAL_MODE:
                session = boto3.Session(profile_name=profile_name)
            else:
                session = boto3.Session(
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name="eu-west-2",  # or your AWS region
                )

            s3_client = session.client("s3")

            # Create the report
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data, current_data=self.current_data
            )

            # Use BytesIO to get the report HTML
            with StringIO() as buffer:
                report.save_html(buffer)
                buffer.seek(0)
                report_html = buffer.read()

            # Save the report directly to S3
            logger_config.logger.info(
                f"Saving drift report to {bucket_name}/{object_name}"
            )
            s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=report_html)
            logger_config.logger.info(
                f"Saved drift report to {bucket_name}/{object_name}"
            )
            return report.as_dict()

        except NoCredentialsError as e:
            logger_config.logger.error(
                f"Error saving drift report to {bucket_name}/{object_name}:{e}"
            )
            raise NoCredentialsError(
                "Credentials not available. Make sure the profile "
                "name is correct and the credentials are set up properly."
            )
        except PartialCredentialsError as e:
            logger_config.logger.error(
                f"Error saving drift report to {bucket_name}/{object_name}:{e}"
            )
            raise PartialCredentialsError(
                "Incomplete credentials. Please check your AWS configuration."
            )
        except Boto3Error as e:
            logger_config.logger.error(
                f"Error saving drift report to {bucket_name}/{object_name}:{e}"
            )
            raise RuntimeError(f"An error occurred with AWS: {str(e)}")

    def check_data_drift(
        self,
        report_location: str,
        bucket_name: str = constants.S3_BUCKET,
    ) -> None:
        """
        Check for data drift in the current data against the reference data.
        Print the result and raise a ValueError if drifted columns are found.

        Args:
            report_location (str) : The drift report file to be checked.
            bucket_name (str): The S3 bucket to store the report.
        """
        data_dict = self.create_report(
            object_name=report_location, bucket_name=bucket_name
        )
        data_drift_metric_result = None

        for metric in data_dict["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                data_drift_metric_result = metric["result"]
                break

        if data_drift_metric_result:
            print(data_drift_metric_result)
        else:
            print("DataDriftMetric not found")

        if (
            data_drift_metric_result
            and data_drift_metric_result["number_of_drifted_columns"] > 0
        ):
            email_functions.send_email(
                "data_drift",
                constants.S3_BUCKET,
                report_location,
                constants.DRIFT_TOPIC,
            )
            return None
