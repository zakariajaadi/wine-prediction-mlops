from pathlib import Path

import mlflow
import pandas as pd
from dotenv import dotenv_values
from prefect import flow
from prefect.logging import get_run_logger
from sqlalchemy import create_engine

from config import read_config
from data_extraction import extract_data
from data_preprocessing import pre_processing
from db import prepare_pipeline_databases
from model_deploy import get_best_run_model_version, promote_model_version_to_champion
from model_monitoring import calculate_drift_metrics, save_drift_metrics_to_db, update_drift_processed_flags
from model_train import set_mlflow_environment, train_hyperparameters_tuning
from utils import download_df_from_minio

# ----- App Flows ---- #
@flow(name="wine_quality_ml_pipeline")
def ml_workflow():
    """
    Train and Automatic Deployment Prefect Flow.
    """
    # Fetch config
    config = read_config()

    # Prepare databases (for monitoring, mlflow ...)
    db_names_list = [config.database.monitoring_db_name,
                     config.database.mlflow_db_name,
                     conf.database.prefect_db_name]

    prepare_pipeline_databases(config, db_names_list)

    # Data extraction
    raw_data_df=extract_data(config)

    # Pre-processing
    train_x, test_x, train_y, test_y= pre_processing(config,raw_data_df)

    # Training & Hyperparameter tuning & mlflow tracking
    set_mlflow_environment(config)
    best_result= train_hyperparameters_tuning(config,train_x,test_x,train_y,test_y)

    # Automatic deployment
    best_model_version = get_best_run_model_version(config)
    promote_model_version_to_champion(config, best_model_version)

    # Model is served as a long-running docker service using FastAPI (check docker-compose file)


@flow(name="wine_quality_monitoring_pipeline")
def monitoring_workflow():
    """
    Monitoring Prefect Flow.
    """
    # Get Prefect logger
    logger=get_run_logger()

    # Fetch config
    conf = read_config()
    data_bucket= conf.data.bucket
    train_data_key, test_data_key= conf.data.train_data_key, conf.data.test_data_key
    target_name= conf.data.target_name
    model_name= conf.mlflow.registered_model_name
    tracking_uri= conf.mlflow.tracking_uri
    monitoring_db_uri= conf.database.construct_db_uri(conf.database.monitoring_db_name)

    # Ensure pipeline databases are available (for monitoring, mlflow ...)
    db_names_list = [conf.database.monitoring_db_name,
                     conf.database.mlflow_db_name,
                     conf.database.prefect_db_name]

    prepare_pipeline_databases(conf, db_names_list)

    # Set mlflow tracking uri
    mlflow.set_tracking_uri(tracking_uri)

    # Load reference data (train data only)
    reference_features_df = download_df_from_minio(data_bucket, train_data_key)
    reference_features_df = reference_features_df.drop([target_name], axis=1)

    # Load production inference data from postgres
    table_name = f"inference_{model_name}"
    engine=create_engine(monitoring_db_uri)
    inference_df = pd.read_sql_table(table_name, con=engine)
    inference_features_df=inference_df[reference_features_df.columns.to_list()]

    # Check if there are enough predictions since last drift
    min_drift_samples = 5
    new_samples_number=inference_df.query("drift_processed == False").shape[0]
    has_sufficient_samples = new_samples_number >= min_drift_samples

    if has_sufficient_samples:

        # Calculate drift
        metrics_dict=calculate_drift_metrics(conf, reference_features_df, inference_features_df)

        # Save metrics
        save_drift_metrics_to_db(conf, metrics_dict)

        # Set drift_processed flag to True
        table_name = f"inference_{model_name}"
        update_drift_processed_flags(conf, table_name)

    else:

        logger.info(f"Not enough samples accumulated for drift calculation.")


if __name__ == "__main__":
    # ----- Deploy & Schedule Flows ---- #

    # Get env dict from .env
    env_vars=dotenv_values()
    env_vars["PREFECT_API_URL"] = "http://prefect:4200/api"

    # Deploy ml pipeline flow
    ml_workflow.deploy(
       name="wine_quality_ml_pipeline_production",
       cron="0 0 * * 0", #every sunday midnight
       work_pool_name="my_k8s_pool",
       image="zakariajaadi/k8s-getting-started:1.0.0",
       job_variables={
           "image_pull_policy": "Always",
           "env": env_vars
       },
       build=False,
       push=False
    )

    # Deploy monitoring  flow
    monitoring_workflow.deploy(
        name="wine_quality_monitoring_production",
        work_pool_name="my_k8s_pool",
        image="zakariajaadi/k8s-getting-started:1.0.0",
        job_variables={
            "image_pull_policy": "Always",
            "env": env_vars
        },
        cron="0 12 * * *", # Every day 12pm
        build=False,
        push=False
    )









