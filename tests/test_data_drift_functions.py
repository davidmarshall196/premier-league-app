import pytest
import pandas as pd
from unittest.mock import patch, Mock
try:
    from premier_league import data_drift_functions
except ImportError:
    import data_drift_functions

def test_init():
    ref_data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    curr_data = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    detector = data_drift_functions.DriftDetector(ref_data, curr_data)

    assert detector.reference_data.equals(ref_data)
    assert detector.current_data.equals(curr_data)

@patch('premier_league.data_drift_functions.Report')
def test_create_report(MockedReport):
    ref_data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    curr_data = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    detector = data_drift_functions.DriftDetector(ref_data, curr_data)

    mocked_report = Mock()
    MockedReport.return_value = mocked_report

    report_dict = {'metrics': []}
    mocked_report.as_dict.return_value = report_dict

    assert detector.create_report("Mock Object Name") == report_dict

@patch('premier_league.data_drift_functions.Report') 
def test_check_data_drift_no_drift(MockedReport):
    ref_data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    curr_data = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    detector = data_drift_functions.DriftDetector(ref_data, curr_data)

    mocked_report = Mock()
    MockedReport.return_value = mocked_report
    mocked_report.as_dict.return_value = {'metrics': [{'metric': 'DatasetDriftMetric', 'result': {'number_of_drifted_columns': 0}}]}

    detector.check_data_drift("Mock Object Name")
