import time

import mlflow
import pandas as pd
import requests
from dotenv import dotenv_values
from prefect import flow
from prefect.logging import get_run_logger
from sqlalchemy import create_engine

from config import read_config
from data_extraction import extract_data
from data_preprocessing import pre_processing
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

    # Model is served as a long-running  service using FastAPI (check model manifest under k8s/model)


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

@flow(name="wine_quality_inference_simulation_pipeline")
def inference_simulation_workflow(batch_size=30):
    """
    Monitoring Prefect Flow.
    """
    # Get Prefect logger
    logger=get_run_logger()

    # Fetch conf
    conf = read_config()
    tracking_uri = conf.mlflow.tracking_uri
    data_bucket = conf.data.bucket
    train_data_key, test_data_key = conf.data.train_data_key, conf.data.test_data_key
    target_name = conf.data.target_name
    model_name = conf.mlflow.registered_model_name
    model_endpoint_url=conf.mlflow.model_endpoint_url

    # Set mlflow environment
    mlflow.set_tracking_uri(tracking_uri)

    # Load reference & simulation (test) data
    reference_df = download_df_from_minio(data_bucket, train_data_key).drop(columns=[target_name])
    simulation_df = download_df_from_minio(data_bucket, test_data_key).drop(columns=[target_name])

    # Iterate over simulation df in batches
    for i in range(0, len(simulation_df), batch_size):

        # Prepare request payload with batch rows
        simulation_batch_df = simulation_df[i:i + batch_size]
        payload_dict = {"inputs": simulation_batch_df.to_dict(orient="records")}

        # Send request & Get predictions
        headers = {'Content-Type': 'application/json'}
        response = requests.post(model_endpoint_url, headers=headers, json=payload_dict)

        if response.status_code == 200:

            logger.info(f" Batch {i // batch_size + 1}: Successfully processed {len(simulation_batch_df)} samples")

            # Task 2: Calculate drift metrics
            metrics_dict = calculate_drift_metrics(conf, reference_df, simulation_batch_df)

            # Task 3: Save metrics
            save_drift_metrics_to_db(conf, metrics_dict)

            # Task 4: Flag drift precessed inference data
            table_name = f"inference_{model_name}"
            update_drift_processed_flags(conf, table_name)

        else:
            logger.warning(f"Batch {i // batch_size + 1}: Processing failed with status code {response.status_code}")

        time.sleep(2)












