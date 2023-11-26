import json
import sys
from io import StringIO

import great_expectations as ge
import numpy as np
import pandas as pd
import pytest

try:
    from premier_league import expectations_helpers
except ImportError:
    import expectations_helpers

@pytest.fixture
def load_sample_data():
    d_string = """
        str_col int_col miss_col
            A   1       8.2
            A   4       None
            A   0       2.99
            B   8       None
            B   3       0.29
            B   3       1.10
    """
    data_sample = pd.read_csv(StringIO(d_string), sep=r"\s+", index_col=False)
    data_sample = data_sample.replace("None", np.NaN)
    data_sample["miss_col"] = pd.to_numeric(data_sample["miss_col"], errors="coerce")
    return data_sample
  
def test_column_type(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    assert ge_object._column_type("str_col") == "str"
    assert ge_object._column_type("int_col") == "int64"
    assert ge_object._column_type("miss_col") == "float64"

def test_missing_fraction(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    assert ge_object._missing_fraction("miss_col", 10) == 0.233


def test_min_value(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    assert ge_object._min_value("int_col", 10) == -0.8


def test_max_value(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    assert ge_object._max_value("int_col", 20) == 9.6

def test_add_max_min_expectations(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    data_ge = ge.from_pandas(load_sample_data)
    ge_output = ge_object._add_max_min_expectations(data_ge, "int_col", 10, 10)
    suite = json.dumps(ge_output.get_expectation_suite().to_json_dict(), sort_keys=True)
    test_suite = """{"data_asset_type": "Dataset", "expectation_suite_name": "default", "expectations": [{"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "int_col", "max_value": 8.8, "min_value": -0.8}, "meta": {}}], "ge_cloud_id": null, "meta": {"great_expectations_version": "0.18.3"}}"""
    assert suite == test_suite

def test_add_cat_expectations(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    data_ge = ge.from_pandas(load_sample_data)
    ge_output = ge_object._add_cat_expectations(data_ge, "str_col", 10)
    suite = json.dumps(ge_output.get_expectation_suite().to_json_dict(), sort_keys=True)
    test_suite = """{"data_asset_type": "Dataset", "expectation_suite_name": "default", "expectations": [{"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"column": "str_col", "value_set": ["A", "B"]}, "meta": {}}], "ge_cloud_id": null, "meta": {"great_expectations_version": "0.18.3"}}"""
    assert suite == test_suite

def test_generate_expectations(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    ge_output = ge_object.generate_expectations()
    suite = json.dumps(ge_output.get_expectation_suite().to_json_dict(), sort_keys=True)
    test_suite = """{"data_asset_type": "Dataset", "expectation_suite_name": "default", "expectations": [{"expectation_type": "expect_table_columns_to_match_set", "kwargs": {"column_set": ["str_col", "int_col", "miss_col"]}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "str_col", "type_": "str"}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_null", "kwargs": {"column": "str_col", "mostly": 0}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"column": "str_col", "value_set": ["A", "B"]}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "int_col", "type_": "int64"}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_null", "kwargs": {"column": "int_col", "mostly": 0}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "int_col", "max_value": 8.8, "min_value": -0.8}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_of_type", "kwargs": {"column": "miss_col", "type_": "float64"}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_null", "kwargs": {"column": "miss_col", "mostly": 0.233}, "meta": {}}, {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "miss_col", "max_value": 8.991, "min_value": -0.5009999999999999}, "meta": {}}], "ge_cloud_id": null, "meta": {"great_expectations_version": "0.18.3"}}"""
    assert suite == test_suite

def test_view_full_suite(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    data_ge = ge.from_pandas(load_sample_data)
    ge_object._add_cat_expectations(data_ge, "str_col", 10)
    suite = json.dumps(
        expectations_helpers.view_full_suite(
            data_ge).to_json_dict(), sort_keys=True)
    test_suite = """{"data_asset_type": "Dataset", "expectation_suite_name": "default", "expectations": [{"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"column": "str_col", "value_set": ["A", "B"]}, "meta": {}}], "ge_cloud_id": null, "meta": {"great_expectations_version": "0.18.3"}}"""
    assert suite == test_suite

def test_view_suite_summary(load_sample_data):
    ge_object = expectations_helpers.AutoGreatExpectations(
        load_sample_data)
    data_ge = ge.from_pandas(load_sample_data)
    ge_object._add_cat_expectations(data_ge, "str_col", 10)
    captured_output = StringIO()
    sys.stdout = captured_output
    expectations_helpers.view_suite_summary(data_ge)
    sys.stdout = sys.__stdout__
    test_summary = (
        ""
    )
    assert captured_output.getvalue() == test_summary
