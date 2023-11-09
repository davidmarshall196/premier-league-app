"""Functions related to evaluating models."""
from sklearn import metrics


def evaluate_model(
    predictions: list, 
    actual_values: list,
    classification: bool = True
) -> dict:
    """Evaluate a model on test data.

    Args:
        predictions (list): List of predictions made by the model.
        actual_values (list): Actual values that the model was attempting to predict.

    Returns:
        evaluation_metrics (dict): Key-value pairs of metric names and values.
    """
    evaluation_metrics = {}
    if classification:
        evaluation_metrics["mcc"] = metrics.matthews_corrcoef(actual_values, predictions)
        evaluation_metrics["accuracy"] = metrics.accuracy_score(actual_values, predictions)
        evaluation_metrics["f1"] = metrics.f1_score(actual_values, predictions,
                                               average='weighted')
        evaluation_metrics["confusion_matrix"] = metrics.confusion_matrix(
            actual_values, predictions
        ).tolist()
    else:
        evaluation_metrics["r2_score"] = metrics.r2_score(actual_values, predictions)
        evaluation_metrics["median_ae"] = metrics.median_absolute_error(actual_values, predictions)
        evaluation_metrics["mean_ae"] = metrics.mean_absolute_error(actual_values, predictions)
    return evaluation_metrics
