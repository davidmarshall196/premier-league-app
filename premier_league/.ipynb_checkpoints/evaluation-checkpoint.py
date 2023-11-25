"""Functions related to evaluating models."""
from sklearn import metrics
import mlflow

# import constants
try:
    from premier_league import (
        logger_config
    )
except ImportError:
    import logger_config

def evaluate_model(
    predictions: list, 
    actual_values: list,
    model_type: str = 'result',
    run_id: str = None
) -> dict:
    """Evaluate a model on test data.

    Args:
    - predictions (list): List of predictions made by the model.
    - actual_values (list): Actual values that the model was attempting to predict.

    Returns:
    - evaluation_metrics (dict): Key-value pairs of metric names and values.
    """
    logger_config.logger.info(
        f'Evaluating {model_type} model'
    )
    if model_type.lower() not in ['result', 'home', 'away']:
        raise ValueError('Model type should be "result", "home" or "away".')
    
    evaluation_metrics = {}
    if model_type == 'result':
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
        
    with mlflow.start_run(run_id=run_id) as run:
        logger_config.logger.info(
            "logging evaluation metrics to mlflow"
        )
        for metric, value in evaluation_metrics.items():
            if metric != "confusion_matrix":  
                mlflow.log_metric(metric, value)
    return evaluation_metrics