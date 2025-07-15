import time

import mlflow
import pandas as pd
import requests
from tqdm import tqdm

from config import read_config, AppConfig
from db import prepare_pipeline_databases
from model_monitoring import calculate_drift_metrics, save_drift_metrics_to_db, update_drift_processed_flags
from utils import download_df_from_minio


def simulate_production(conf:AppConfig,reference_df,simulation_df,batch_size):
    """
     Simulate production-level data inference.

    """

    # Fetch conf
    model_name = conf.mlflow.registered_model_name

    # Iterate over simulation df in batches
    for i in tqdm(range(0, len(simulation_df), batch_size), desc="Processing Batches"):

        # Prepare request payload from batch
        simulation_batch_df=simulation_df[i:i+batch_size]
        payload_dict={"inputs": simulation_batch_df.to_dict(orient="records")}

        # Send request & Get predictions
        api_url = "http://localhost:30080/predict"
        headers = {'Content-Type': 'application/json'}
        response=requests.post(api_url,headers=headers, json=payload_dict)

        if response.status_code == 200 :

            tqdm.write(f" Batch {i // batch_size + 1}: Successfully processed {len(simulation_batch_df)} samples")

            # Calculate drift metrics
            metrics_dict = calculate_drift_metrics(conf, reference_df, simulation_batch_df)

            # Save metrics
            save_drift_metrics_to_db(conf, metrics_dict)

            # Flag drift precessed inference data
            table_name = f"inference_{model_name}"
            update_drift_processed_flags(conf, table_name)

        else:
            tqdm.write( f"Batch {i // batch_size + 1}: Processing failed with status code {response.status_code}")

        time.sleep(5)


def main(batch_size:int =30):

        # Fetch conf
        conf = read_config()
        tracking_uri = conf.mlflow.tracking_uri
        data_bucket = conf.data.bucket
        train_data_key, test_data_key = conf.data.train_data_key, conf.data.test_data_key
        target_name = conf.data.target_name

        mlflow.set_tracking_uri(tracking_uri)

        # Ensure pipeline databases are available (for monitoring, mlflow ...)
        db_names_list = [conf.database.monitoring_db_name,
                         conf.database.mlflow_db_name,
                         conf.database.prefect_db_name]
        prepare_pipeline_databases(conf, db_names_list)

        # Load reference data
        reference_df = download_df_from_minio(data_bucket, train_data_key)
        reference_df = reference_df.drop([target_name], axis=1)

        # Load test data (used to simulate production data)
        production_df = download_df_from_minio(data_bucket, train_data_key)
        production_df = production_df.drop([target_name], axis=1)

        # Use test data for production simulation
        simulate_production(conf,reference_df,production_df,batch_size)

if __name__ == "__main__":
    import os

    # ENV_MODE env var controls which model to use for simulation (Dev or Prod version)
    os.environ["ENV_MODE"] = "prod"
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = "30432"
    os.environ["MINIO_ENDPOINT"] = "http://localhost:31248"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:30500"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:31248"

    main()








