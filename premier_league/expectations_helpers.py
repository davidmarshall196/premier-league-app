"""Functions to help with Great Expectations."""
import json

import great_expectations as ge
import pandas as pd


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

    def _add_max_min_expectations(self, 
                                  ge_object, 
                                  col, 
                                  min_buffer, 
                                  max_buffer,
                                  verbose=True):
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
                print(f"Adding min/max expecations to column {col}")
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
                print(f"Adding categorical expecations to column {col}")
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
                print(f"Adding categorical expecations to column {col}")
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
        verbose: bool = True
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
                    print(f"Adding missing expectations to column {col}")
                data_ge.expect_column_values_to_be_null(
                    col, self._missing_fraction(col, buffer=self.missing_buffer)
                )
            if expect_min_max:
                data_ge = self._add_max_min_expectations(
                    data_ge, col, min_buffer=self.min_buffer, 
                    max_buffer=self.max_buffer,
                    verbose=verbose
                )
            if expect_cat_vars:
                data_ge = self._add_cat_expectations(
                    data_ge, 
                    col, 
                    thresh=self.categorical_threshold,
                    verbose=verbose
                )
        print("Done")
        self.data_ge = data_ge
        return self.data_ge


def save_expectations(data_ge, expectations_path):
    """Save expectations locally as a json file.

    Args:
        data_ge (expectations object): A Great Expectations object
        from the Great Expectations library.
        expectations_path (string): Full path of the saved file.

    Returns:
        None.
    """
    with open(expectations_path, "w") as expectations_file:
        expectations_file.write(
            json.dumps(
                data_ge.get_expectation_suite(
                    discard_failed_expectations=False
                ).to_json_dict()
            )
        )
    print(f"Expectations saved at {expectations_path}")


def validate_data(data: pd.DataFrame, data_expectations: dict) -> dict:
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
        print(validation_results["statistics"])
    else:
        print("Validated:", validation_results["success"])
        print(validation_results["statistics"])
        for result in validation_results["results"]:
            if not result["success"]:
                print(result)
        raise Exception("Data does not meet expectations!")
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
    print(f"Total Expectations: {total_exp}")
    distinct_list = set(
        [
            s
            for s in suite_str.replace('"', "").replace(",", "").split()
            if "expect_" in s
        ]
    )
    print("Counts:")
    for exp in distinct_list:
        exp_count = suite_str.count(exp)
        print(f"{exp}: {exp_count}")


def load_expectations(expectations_path: str):
    """Load expectations saved locally from a json file.

    Args:
        expectations_path (string): Full path of the file containing
                                    the expectations.

    Returns:
        data_expectations (expectations object): JSON of expectations.
    """
    with open(expectations_path, "r") as json_file:
        data_expectations = json.load(json_file)
    print(f"Expectations loaded from {expectations_path}")
    return data_expectations


def view_full_suite(data_ge):
    """Prints a all the current expectations.

    Args:
        data_ge (expectations object): A Great Expectations object
        from the Great Expectations library.

    Returns:
        suite (expectations object): A json list of current expectations.
    """
    print(f"Data GE object type: {type(data_ge)}")
    suite = data_ge.get_expectation_suite(discard_failed_expectations=False)
    return suite