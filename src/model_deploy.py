from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from prefect import task
from prefect.logging import get_run_logger

from config import read_config, AppConfig


@task(name="Get best model version")
def get_best_run_model_version(conf: AppConfig):
    """
    Search runs for the best model (lowest rmse) and returns it's version.

    Args:
        conf (AppConfig) : Config object
    Returns:
        model_version (str):  model version of the best run

    """
    # Get prefect logger
    logger = get_run_logger()

    # Fetch config
    model_name = conf.mlflow.registered_model_name
    experiment_name = conf.mlflow.experiment_name

    # Get runs in asc order of rmse
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow_client = MlflowClient()
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id],
                                     order_by=["metrics.rmse ASC"],
                                     run_view_type=ViewType.ACTIVE_ONLY)

    # Get best run (lowest rmse)
    best_run = runs[0]
    best_run_id = best_run.info.run_id

    # Fetch model version of the best run
    model_versions_list = mlflow.search_model_versions(
            max_results=1, filter_string=f"name='{model_name}' AND run_id='{best_run_id}'"
    )
    best_model_version = model_versions_list[0].version

    return best_model_version

@task(name="promote model to champion")
def promote_model_version_to_champion(conf: AppConfig, new_model_version: str):
    """
    Sets a specific model version as the champion using MLflow aliases.

    Args:
        conf (AppConfig): Configuration object containing MLflow settings
        new_model_version (str): Registered model version to promote as champion

    Returns:
        None
    """

    # Get prefect logger
    logger = get_run_logger()

    mlflow_client = MlflowClient()

    # Fetch model name from config
    model_name = conf.mlflow.registered_model_name


    current_champion_version = None
    try:
        # Try to fetch current champion version
        current_champion_version = mlflow_client.get_model_version_by_alias(
            name=model_name, alias="champion"
        ).version

    except MlflowException:
        logger.info("This is likely the first promotion, no champion model exists.")
        pass

    try:
        # Only transition the current champion to 'previous' if it's different from the new version
        if current_champion_version and current_champion_version != new_model_version:

            mlflow_client.set_registered_model_alias(
                name=model_name, version=current_champion_version, alias="previous"
            )

            logger.info(f"Model {model_name} version {current_champion_version} alias transitioned to 'previous'.")

        # If the new model version is not already set as champion, promote it
        if current_champion_version != new_model_version:

            mlflow_client.set_registered_model_alias(
                name=model_name, version=str(new_model_version), alias="champion"
            )

            logger.info(f"Model {model_name} version {new_model_version} is now set as champion.")
        else:
            logger.info(f"Model {model_name} version {new_model_version} is already set as champion.")

    except MlflowException as e:
        logger.exception(f"Failed to promote model version {new_model_version} to champion: {e}")
        raise



def main():

    # Fetch conf
    conf = read_config()
    tracking_uri = conf.mlflow.tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    model_version=get_best_run_model_version(conf)
    promote_model_version_to_champion(conf, model_version)


if __name__ == "__main__":
    main()
