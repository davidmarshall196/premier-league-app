from typing import List
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
import pandas as pd

# import constants
try:
    from premier_league import config
    from premier_league import constants
except ModuleNotFoundError:
    import config
    import constants


def open_mlflow_tracking(experiment_name=constants.EXP_NAME):
    db_uri = f"postgresql+psycopg2://postgres:{config.RDS_DB_PASSWORD}@{config.RDS_ENDPOINT}:5432/{config.RDS_DB_ID}" 

    mlflow.set_tracking_uri(db_uri)    
    mlflow.set_experiment(experiment_name)


def get_all_experiments(client: MlflowClient) -> List[dict]:
    """
    Retrieve all experiments from the MLflow tracking server.

    Parameters:
    - client (MlflowClient): An instance of the MLflowClient.

    Returns:
    List[dict]: A list of experiments, each represented as a dictionary.
    """
    experiments = client.search_experiments()
    return [{"name": exp.name, "experiment_id": exp.experiment_id} for exp in experiments]

def get_runs_from_experiment(client: MlflowClient, experiment_id: str) -> List[Run]:
    """
    Get all runs from a specified experiment.

    Parameters:
    - client (MlflowClient): An instance of the MLflowClient.
    - experiment_id (str): The experiment ID from which to retrieve runs.

    Returns:
    List[Run]: A list of Run objects containing run details.
    """
    return client.search_runs([experiment_id])


def print_run_details(run: Run) -> None:
    """
    Print the details of a given run, including parameters, metrics, and tags.

    Parameters:
    - run (Run): An MLflow Run object.

    Returns:
    None
    """
    print(f"Run ID: {run.info.run_id}")
    print("Parameters:")
    for key, value in run.data.params.items():
        print(f"  {key}: {value}")
    print("Metrics:")
    for key, value in run.data.metrics.items():
        print(f"  {key}: {value}")
    print("Tags:")
    for key, value in run.data.tags.items():
        print(f"  {key}: {value}")
    print("-" * 30)


def runs_to_dataframe(runs: List[Run]) -> pd.DataFrame:
    """
    Convert a list of MLflow runs into a pandas DataFrame.

    Parameters:
    - runs (List[Run]): A list of MLflow Run objects.

    Returns:
    pd.DataFrame: A pandas DataFrame where each row represents a run and columns
    represent run details such as parameters, metrics, and tags.
    """
    runs_data = []

    for run in runs:
        # Extract run information
        run_id = run.info.run_id
        parameters = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags
        
        # Combine parameters, metrics, and tags into a single dictionary
        run_info = {**parameters, **metrics, **tags, "run_id": run_id}
        runs_data.append(run_info)
    
    # Create a DataFrame from the combined run information
    return pd.DataFrame(runs_data)

def get_runs_from_experiment(client: MlflowClient, experiment_id: str) -> List[Run]:
    """
    Get all runs from a specified experiment.

    Parameters:
    - client (MlflowClient): An instance of the MLflowClient.
    - experiment_id (str): The experiment ID from which to retrieve runs.

    Returns:
    List[Run]: A list of Run objects containing run details.
    """
    runs = client.search_runs(experiment_ids=[experiment_id])
    return runs