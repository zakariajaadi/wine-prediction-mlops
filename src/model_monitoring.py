import datetime
import os

import mlflow
import pandas as pd
from prefect import task
from prefect.logging import get_run_logger
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.exc import SQLAlchemyError

from config import read_config, AppConfig
from utils import calculate_feature_drift, calculate_prediction_drift


@task(name="Calculate Drift Metrics")
def calculate_drift_metrics(conf: AppConfig, reference_df: pd.DataFrame, current_df: pd.DataFrame, model=None):
    """Calculates features and prediction drift metrics"""

    # Get prefect logger
    logger = get_run_logger()

    if model is None:
        model = mlflow.sklearn.load_model(conf.mlflow.model_uri)

    # Calculate drift using DeepChecks
    agg_features_drift_score = calculate_feature_drift(reference_df, current_df, model)
    prediction_drift_score = calculate_prediction_drift(reference_df, current_df, model)

    logger.info("Drift metrics calculated.")

    return {
        "agg_features_drift_score": agg_features_drift_score,
        "prediction_drift_score": prediction_drift_score
    }


@task(name="Save Drift Metrics")
def save_drift_metrics_to_db(conf: AppConfig, metrics_dict: dict):
    """Save calculated drift metrics into the monitoring database."""
    # Get prefect logger
    logger = get_run_logger()

    # Fetch conf
    model_name = conf.mlflow.registered_model_name
    monitoring_db_name = conf.database.monitoring_db_name
    monitoring_db_uri = conf.database.construct_db_uri(monitoring_db_name)

    table_name="model_metrics"

    # Convert metrics to DataFrame
    df = pd.DataFrame([{
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "registered_model_name": model_name,
        "agg_features_drift_score": metrics_dict["agg_features_drift_score"],
        "prediction_drift_score": metrics_dict["prediction_drift_score"]
    }])

    # Save metrics to predictions_metrics table in the monitoring DB
    try:
        engine = create_engine(monitoring_db_uri)
        df.to_sql(table_name, engine, if_exists="append", index=False)
        logger.info(f"Drift metrics saved to the table '{table_name}' in database '{monitoring_db_name}'.")

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error occurred while saving drift metrics: {e}")
        raise


@task(name="Flag Drift Processed Data")
def update_drift_processed_flags(conf, table_name: str):
    # Get prefect logger
    logger = get_run_logger()

    # Fetch db config
    monitoring_db_name = conf.database.monitoring_db_name
    monitoring_db_uri = conf.database.construct_db_uri(monitoring_db_name)

    # SQLAlchemy engine
    engine = create_engine(monitoring_db_uri)

    # Reflect the table from the database
    table = Table(table_name, MetaData(), autoload_with=engine)

    # Update drift_processed flag using SQLAlchemy core
    with engine.begin() as connection:
        try:
            update_query = (table
                            .update()
                            .where(table.c.drift_processed.is_(False))
                            .values(drift_processed=True))
            result = connection.execute(update_query)

            logger.info(f"Successfully updated 'drift_processed' flag to True for {result.rowcount} rows in '{table_name}' table in {monitoring_db_name}")
        except Exception as e:
            logger.error(f"Error updating 'drift_processed' flag for {table_name} in database {monitoring_db_name}: {e}")
            raise


def main():
    # Fetch config
    conf = read_config()
    train_data_path, test_data_path = conf.data.train_data_path, conf.data.test_data_path
    target_name = conf.data.target_name

    # Set Tracking Uri
    mlflow.set_tracking_uri(conf.mlflow.tracking_uri)

    # Load processed test data
    reference_df = pd.read_csv(train_data_path).drop([target_name], axis=1)
    current_df = pd.read_csv(test_data_path).drop([target_name], axis=1)

    # Induce synthetic drift
    current_df['volatile_acidity'] = current_df['volatile_acidity'] * 2.5

    # Calculate drift
    metrics_dict = calculate_drift_metrics(conf, reference_df, current_df)

    # Save drift
    save_drift_metrics_to_db(conf, metrics_dict)


if __name__ == "__main__":
    print(os.getenv("ENV_MODE", "dev"))
    main()
