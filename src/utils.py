import datetime
import logging
from typing import List
import io

import mlflow
import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, PredictionDrift, MultivariateDrift
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import boto3


from config import read_config

# Get Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#------------- Train and Eval utils -------------#

def evaluation_metrics(actual,pred):
    """
    Evaluation metrics for regression
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#------------- Minio utils -------------#

def upload_df_to_minio(df, bucket: str, destination_key: str):
    """
    Uploads a pandas DataFrame as a CSV file to MinIO/S3.
    """
    # Fetch config
    conf = read_config()
    minio_endpoint = conf.minio.endpoint_url
    access_key = conf.minio.access_key
    secret_key = conf.minio.secret_key

    # Convert DataFrame to CSV bytes
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Create S3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        s3.put_object(Bucket=bucket, Key=destination_key, Body=csv_bytes, ContentType='text/csv')
        logger.info(f"Uploaded DataFrame to s3://{bucket}/{destination_key}")
    except Exception as e:
        logger.exception(f"Failed to upload DataFrame to MinIO: {e}")
        raise

def download_df_from_minio(bucket: str, object_key: str) -> pd.DataFrame:
    """
    Downloads a CSV file from MinIO/S3 and returns it as a pandas DataFrame.
    """
    # Fetch config
    conf = read_config()
    minio_endpoint = conf.minio.endpoint_url
    access_key = conf.minio.access_key
    secret_key = conf.minio.secret_key

    # Create S3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        response = s3.get_object(Bucket=bucket, Key=object_key)
        content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        logger.info(f"Downloaded DataFrame from s3://{bucket}/{object_key}")
        return df
    except Exception as e:
        logger.exception(f"Failed to download CSV from MinIO: {e}")
        raise



#------------- Monitoring utils -------------#

def calculate_feature_drift(reference_df,current_df,model):
    """
    Features Drift calculation using DeepChecks.
    """

    # Initialize Features drift
    ref_ds = Dataset(df=reference_df.to_numpy(), cat_features=[])
    curr_ds = Dataset(df=current_df.to_numpy(), cat_features=[])
    check_feature_drift=FeatureDrift()

    # Run drift calculation
    result=check_feature_drift.run(train_dataset=ref_ds,test_dataset=curr_ds,model=model.named_steps["model"])

    # Reduce Features Drift
    weighted_drift_score=result.reduce_output()['L3 Weighted Drift Score']

    return weighted_drift_score

def calculate_prediction_drift(reference_df,current_df,model):
    """
    Prediction Drift calculation using DeepChecks.
    """
    # Initialize Prediction Drift
    ref_ds = Dataset(df=reference_df.to_numpy(), cat_features=[])
    curr_ds = Dataset(df=current_df.to_numpy(), cat_features=[])
    pred_drift_check = PredictionDrift()

    # Prediction
    y_pred_ref= model.predict(reference_df)
    y_pred_curr= model.predict(current_df)

    # Run Prediction Drift
    result = pred_drift_check.run(train_dataset=ref_ds, y_pred_train=y_pred_ref,
                                  test_dataset=curr_ds, y_pred_test= y_pred_curr,
                                  model=model.named_steps["model"])

    return result.value['Drift score']

def calculate_dataset_drift(reference_df,current_df):
    """
    Whole Dataset Drift calculation using DeepChecks.
    """
    # Initialize Whole Dataset Drift (multivariate)
    ref_ds = Dataset(df=reference_df.to_numpy(), cat_features=[])
    curr_ds = Dataset(df=current_df.to_numpy(), cat_features=[])
    check_ds_drift = MultivariateDrift()

    #Run Whole DataSet Drift
    result = check_ds_drift.run(train_dataset=ref_ds, test_dataset=curr_ds)

    return result.value['domain_classifier_drift_score']


#------------- Model utils -------------#

def make_prediction_from_model(samples_data_df: pd.DataFrame, model_uri: str):
    """Makes a prediction using a trained model from mlflow"""

    # Load model and predict
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Sklearn flavor predict return type is a numpy array
    predictions = loaded_model.predict(samples_data_df)

    return predictions.tolist()


def save_predictions(features_df: pd.DataFrame, predictions_list: list):
    """
    Saves predictions live predictions in monitoring DB
    """
    # Fetch config
    conf =  read_config()
    model_name = conf.mlflow.registered_model_name
    monitoring_db_name=conf.database.monitoring_db_name
    monitoring_db_uri = conf.database.construct_db_uri(monitoring_db_name)

    # Add complementary info
    features_df["prediction"] = predictions_list
    features_df["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
    #features_df["model_version"] = os.getenv("MODEL_VERSION", "unknown")
    features_df["drift_processed"] = False

    # Define dynamic table name
    table_name = f"inference_{model_name}"

    # Save predictions
    try:
        engine = create_engine(monitoring_db_uri)
        features_df.to_sql(table_name, con=engine, if_exists='append', index=False)

        logger.info(f"predictions saved to table: {table_name} in database {monitoring_db_name}")

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error occurred while saving predictions: {e}")
        raise


#------------- Fast API utils -------------#

# Data validation with pydantic
class DataModel(BaseModel):

    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class InputDataModel(BaseModel):
    inputs: List[DataModel]

