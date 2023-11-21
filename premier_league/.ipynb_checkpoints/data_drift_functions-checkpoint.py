from typing import Dict, Any
from pandas import DataFrame
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DriftDetector:
    """
    Class to detect and report data drift between reference and
    current datasets.

    Attributes:
        reference_data (DataFrame): Reference data to compare.
        current_data (DataFrame): Current data to check against reference data.
    """

    def __init__(
        self,
        reference_data: DataFrame,
        current_data: DataFrame
    ):
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
        report_location: str = "drift_report.html"
    ) -> Dict[str, Any]:
        """
        Create a data drift report using Evidently's DataDriftPreset.

        Args:
            report_location (str) : File report will be created to.

        Returns:
            Dict[str, Any]: A dictionary containing the data drift report.
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )
        report.save_html(report_location)
        print(f"HTML report saved at {report_location}")
        return report.as_dict()

    def check_data_drift(
        self,
        report_location: str = "drift_report.html"
    ) -> None:
        """
        Check for data drift in the current data against the reference data.
        Print the result and raise a ValueError if drifted columns are found.

        Args:
            report_location (str) : The drift report file to be checked.
        """
        data_dict = self.create_report(report_location=report_location)
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
            raise ValueError("Columns have drifted. Check report")
