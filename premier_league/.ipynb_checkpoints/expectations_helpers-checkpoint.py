"""Functions to help with Great Expectations."""
import json
import os
import great_expectations as ge
import pandas as pd
from typing import Optional
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from datetime import datetime

# import constants
try:
    from premier_league import constants, logger_config
except ImportError:
    import constants
    import logger_config


class AutoGreatExpectations:
    """For undertaking data validation tasks automatically."""

    def __init__(self, data):
        """Instantiate the auto_great_expectations class.

        Args:
            data (pd.DataFrame): A data frame to calculate the expectations.

        Returns:
            None.
        """
        self.data = data

    def _column_type(self, col: str) -> str:
        """Convert correct format from dtype object.

        Args:
            col (string): A column to which to convert the dtype if necessary.

        Returns:
            c_type (string): The correct column data type.
        """
        self.c_type = str(self.data[col].dtype)
        if self.c_type == "object":
            return "str"
        else:
            return self.c_type

    def _missing_fraction(self, col, buffer):
        """Calculate the missing fraction of a column with custom buffer.

        Args:
            col (string): A column to be used to calculate the missing fraction.
            buffer (int): The percentage buffer for the missingness (e.g. 10 = 10%
                          either side of the current level of missingness).

        Returns:
            frac (float): A missing fraction number. If fraction is
            less than or equal to buffer, returns 0.
        """
        buffer = buffer / 100
        frac = (self.data[col].isnull().sum() / len(self.data[col])).round(3)
        if frac > buffer:
            return frac - buffer
        else:
            return 0

    def _min_value(self, col, min_buffer):
        """Calculate the expected minimum value of a column with custom buffer.

        Args:
            col (string): Column for the minimum expected value calculation.
            buffer (int): The percentage buffer for the min_value (e.g. 10 = 10%
                          lower than the current minimum value).

        Returns:
            val (float): The expected minimum value of a column.
        """
        buffer = min_buffer / 100
        col_diff = (self.data[col].max() - self.data[col].min()) * buffer
        val = self.data[col].min() - col_diff
        return val

    def _max_value(self, col, max_buffer):
        """Calculate the expected maximum value of a column with 20% buffer.

        Args:
            col (string): Column for the maximum expected value calculation.
            buffer (int): The percentage buffer for the max_value (e.g. 10 = 10%
                          higher than the current maximum value).

        Returns:
            val (float): The expected maximum value of a column.
        """
        buffer = max_buffer / 100
        col_diff = (self.data[col].max() - self.data[col].min()) * buffer
        val = self.data[col].max() + col_diff
        return val

    def _add_max_min_expectations(
        self, ge_object, col, min_buffer, max_buffer, verbose=True
    ):
        """Add the minimum and maximum value expectations.

        Args:
            ge_object (GE dataframe): An input GE dataframe.
            col (string): Column for the min and max values to be added.
            buffer (int): The percentage buffer for the max_value (e.g. 10 = 10%
                          higher than the current maximum value).

        Returns:
            ge_object (GE dataframe): The GE dataframe with expectations added.
        """
        if (
            (
                "float"
                in self._column_type(
                    col,
                )
            )
            or (
                "int"
                in self._column_type(
                    col,
                )
            )
        ) and self.data[col].notnull().sum() > 0:
            if verbose:
                logger_config.logger.info(f"Adding min/max expecations to column {col}")
            ge_object.expect_column_values_to_be_between(
                col,
                min_value=self._min_value(col, min_buffer=min_buffer),
                max_value=self._max_value(col, max_buffer=max_buffer),
            )
        return ge_object

    def _add_cat_expectations(self, ge_object, col, thresh, verbose=True):
        """Add categorical set expectations.

        Args:
            ge_object (GE dataframe): An input GE dataframe.
            col (string): Column for the min and max values to be added.
            thresh (int): The threshold for a variable to be included, meaning
                          number of categories. Default=10.

        Returns:
            ge_object (GE dataframe): The GE dataframe with expectations added.
        """
        if (
            "str"
            in self._column_type(
                col,
            )
            and len(self.data[col].value_counts()) <= thresh
        ):
            if verbose:
                logger_config.logger.info(
                    f"Adding categorical expecations to column {col}"
                )
            str_list = list(self.data[col].unique())
            ge_object.expect_column_values_to_be_in_set(col, value_set=str_list)
        if (
            (
                "int"
                in self._column_type(
                    col,
                )
            )
            or (
                "float"
                in self._column_type(
                    col,
                )
            )
        ) and sorted(self.data[col].unique()) == [0, 1]:
            if verbose:
                logger_config.logger.info(
                    f"Adding categorical expecations to column {col}"
                )
            ge_object.expect_column_values_to_be_in_set(col, value_set=[0, 1])
        return ge_object

    def generate_expectations(
        self,
        expect_match_set: bool = True,
        expect_col_types: bool = True,
        expect_missing: bool = True,
        expect_min_max: bool = True,
        expect_cat_vars: bool = True,
        missing_buffer: int = 10,
        min_buffer: int = 10,
        max_buffer: int = 10,
        categorical_threshold: int = 10,
        verbose: bool = True,
    ):
        """Create Great expectations object using input functions.

        Args:
            expect_match_set (bool): Whether to include the column match set
                                     expectations.
            expect_col_types (bool): Whether to include the column types
                                     expectations.
            expect_missing (bool): Whether to include the missing
                                     expectations.
            expect_min_max (bool): Whether to include the min max
                                     expectations.
            expect_cat_vars (bool): Whether to include the categorical set
                                     expectations.
            missing_buffer (int): The missingess buffer - e.g. 10 = 10% buffer
                                  allowed above and below the current level.
            min_buffer (int): The min buffer - e.g. 10 = 10% buffer
                                  allowed below the current level.
            max_buffer (int): The max buffer - e.g. 10 = 10% buffer
                                  allowed above the current level.
            categorical_threshold (int): The threshold to include categorical
                                         variables (number of categories).

        Returns:
            ge_object (GE dataframe): The GE dataframe with expectations added.
        """
        self.missing_buffer = missing_buffer
        self.min_buffer = min_buffer
        self.max_buffer = max_buffer
        self.categorical_threshold = categorical_threshold
        cols = list(self.data.columns)
        data_ge = ge.from_pandas(self.data)
        if expect_match_set:
            data_ge.expect_table_columns_to_match_set(cols)
        print("Generating expectations")
        for col in cols:
            if expect_col_types:
                data_ge.expect_column_values_to_be_of_type(col, self._column_type(col))
            if expect_missing:
                if verbose:
                    logger_config.logger.info(
                        f"Adding missing expectations to column {col}"
                    )
                data_ge.expect_column_values_to_be_null(
                    col, self._missing_fraction(col, buffer=self.missing_buffer)
                )
            if expect_min_max:
                data_ge = self._add_max_min_expectations(
                    data_ge,
                    col,
                    min_buffer=self.min_buffer,
                    max_buffer=self.max_buffer,
                    verbose=verbose,
                )
            if expect_cat_vars:
                data_ge = self._add_cat_expectations(
                    data_ge, col, thresh=self.categorical_threshold, verbose=verbose
                )
        print("Done")
        self.data_ge = data_ge
        return self.data_ge


def save_expectations(
    data_ge,
    expectations_path: str,
    bucket: str = constants.S3_BUCKET,
    validation_results: bool = False,
    profile_name: Optional[str] = "premier-league-app",
):
    """Save expectations locally or to s3.

    Args:
        data_ge (expectations object): A Great Expectations object
        from the Great Expectations library.
        expectations_path (string): Full path of the saved file.
        save_to_s3: Whether to save the file to an S3 Bucket.
        s3_bucket: S3 bucket to save file to.

    Returns:
        None.
    """
    if not validation_results:
        json_file = json.dumps(
            data_ge.get_expectation_suite(
                discard_failed_expectations=False
            ).to_json_dict()
        )
    else:
        json_file = json.dumps(data_ge.to_json_dict())
    if bucket:
        try:
            if constants.LOCAL_MODE:
                session = boto3.Session(profile_name=profile_name)
            else:
                session = boto3.Session(
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name="eu-west-2",
                )
        except NoCredentialsError as e:
            logger_config.logger.error(
                f"An error occurred reading {bucket}/{expectations_path}: %s", str(e)
            )
            raise NoCredentialsError(
                """Credentials not available. Make sure the profile
                name is correct and the credentials are set up properly."""
            )
        except PartialCredentialsError as e:
            logger_config.logger.error(
                f"An error occurred reading {bucket}/{expectations_path}: %s", str(e)
            )
            raise PartialCredentialsError(
                "Incomplete credentials. Please check your AWS configuration."
            )
        except Exception as e:
            logger_config.logger.error(
                f"An error occurred reading {bucket}/{expectations_path}: %s", str(e)
            )
            raise Exception(f"An unexpected error occurred: {str(e)}")
        logger_config.logger.info(
            f"Saving expectations to {bucket}/{expectations_path}"
        )
        s3 = session.client("s3")  # Create a connection to S3
        s3.put_object(Body=json_file, Bucket=bucket, Key=expectations_path)
        logger_config.logger.info(f"Saved expectations to {bucket}/{expectations_path}")
    else:
        with open(expectations_path, "w") as expectations_file:
            expectations_file.write(json_file)


def validate_data(
    data: pd.DataFrame, data_expectations: dict, expectations_path: str
) -> dict:
    """Provide a summary of the validation results.

    Args:
        data (pd.DataFrame): A pandas dataframe to be processed.
        data_expectations (dict): A dictionary of data expectations.

    Returns:
        validation_results (json dict): A summary of the result of the
        expectations.
    """
    data = ge.from_pandas(data, expectation_suite=data_expectations)
    validation_results = data.validate()
    if validation_results["success"]:
        logger_config.logger.info(validation_results["statistics"])
    else:
        logger_config.logger.error(f"Validated: {validation_results['success']}")
        logger_config.logger.error(validation_results["statistics"])
        for result in validation_results["results"]:
            if not result["success"]:
                logger_config.logger.error(result)
        raise Exception("Data does not meet expectations!")
    save_expectations(
        validation_results,
        expectations_path=expectations_path,
        bucket=constants.S3_BUCKET,
        validation_results=True,
    )

    return validation_results.to_json_dict()


def view_suite_summary(data_ge):
    """Prints a summary of the current expectations.

    Args:
        data_ge (expectations object): A Great Expectations object
        from the Great Expectations library.

    Returns:
        None.
    """
    suite = data_ge.get_expectation_suite(discard_failed_expectations=False)
    suite_str = str(suite)
    total_exp = suite_str.count("expectation_type")
    logger_config.logger.info(f"Total Expectations: {total_exp}")
    distinct_list = set(
        [
            s
            for s in suite_str.replace('"', "").replace(",", "").split()
            if "expect_" in s
        ]
    )
    logger_config.logger.info("Counts:")
    for exp in distinct_list:
        exp_count = suite_str.count(exp)
        logger_config.logger.info(f"{exp}: {exp_count}")
        logger_config.logger.info(f"{exp}: {exp_count}")


def latest_exp_file(
    bucket: str = constants.S3_BUCKET,
    prefix: Optional[str] = "app_data/expectations/exp_prem_results",
    profile_name: Optional[str] = "premier-league-app",
):
    try:
        # The profile name is optional and used for local development
        if constants.LOCAL_MODE:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2",
            )
        s3_client = session.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        latest_model = None
        latest_date = None
        logger_config.logger.info("Looking for latest expectations file")
        for obj in response.get("Contents", []):
            filename = obj["Key"]
            file_date_str = filename.split("_")[-1].split(".")[0]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")

        if not latest_date or file_date > latest_date:
            latest_date = file_date
            latest_model = filename

        return latest_model

    except NoCredentialsError:
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError:
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )


def load_latest_expectations(
    expectations_path: str,
    s3_bucket: str = constants.S3_BUCKET,
    profile_name: Optional[str] = "premier-league-app",
):
    """Load latest expectations from S3.

    Args:
        expectations_path (string): Full path of the file containing
                                    the expectations.

    Returns:
        data_expectations (expectations object): JSON of expectations.
    """
    try:
        # The profile name is optional and used for local development
        if constants.LOCAL_MODE:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name="eu-west-2",
            )
        s3_client = session.resource("s3")
        logger_config.logger.info(
            f"Loading expectations from {s3_bucket}/{expectations_path}"
        )
        content_object = s3_client.Object(s3_bucket, expectations_path)
        file_content = content_object.get()["Body"].read().decode("utf-8")
        data_expectations = json.loads(file_content)
        logger_config.logger.info(
            f"Loaded expectations from {s3_bucket}/{expectations_path}"
        )
        return data_expectations

    except NoCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred reading {s3_bucket}/{expectations_path}: %s", str(e)
        )
        raise NoCredentialsError(
            "Credentials not available. Make sure the profile "
            "name is correct and the credentials are set up properly."
        )
    except PartialCredentialsError as e:
        logger_config.logger.error(
            f"An error occurred reading {s3_bucket}/{expectations_path}: %s", str(e)
        )
        raise PartialCredentialsError(
            "Incomplete credentials. Please check your AWS configuration."
        )


def view_full_suite(data_ge):
    """Prints a all the current expectations.

    Args:
        data_ge (expectations object): A Great Expectations object
        from the Great Expectations library.

    Returns:
        suite (expectations object): A json list of current expectations.
    """
    logger_config.logger.info(f"Data GE object type: {type(data_ge)}")
    suite = data_ge.get_expectation_suite(discard_failed_expectations=False)
    return suite
