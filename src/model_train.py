import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from prefect import task
from prefect.logging import get_run_logger
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AppConfig, read_config
from db import prepare_pipeline_databases
from utils import evaluation_metrics


@task(name="mlflow environment setup", retries=3, retry_delay_seconds=5)
def set_mlflow_environment(conf:AppConfig):
    """
    Configures mlflow environment for experiment tracking.

    Args:
          conf (AppConfig) : Configuration object
    Returns:
          None
    """

    # Get prefect logger
    logger=get_run_logger()

    # Fetch config
    tracking_uri=conf.mlflow.tracking_uri
    experiment_name=conf.mlflow.experiment_name

    try :
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    except MlflowException as e:
        logger.exception(f"Failed to set MLflow environment")
        raise

@task(name="Train & Hyper parameters tuning")
def train_hyperparameters_tuning(conf:AppConfig, train_x, test_x, train_y, test_y):
        """
         Performs hyperparameter tuning using HyperOpt to optimize an ElasticNet model

        This function does the following:
            - Scales the features using a scaler.
            - Defines a search space for ElasticNet hyperparameters (`alpha` and `l1_ratio`).
            - Tunes hyperparameters using HyperOpt's to explore only promising regions.
            - Uses RMSE as the objective metric to minimize.
            - Logs parameters, metrics (RMSE, MAE, R2), pipeline as artifacts in mLflow.
        Args:
            conf (AppConfig): Configuration object
            train_x (pd.DataFrame): Training data features
            test_x (pd.DataFrame): Testing data features
            train_y (pd.DataFrame): Training data targets
            test_y (pd.DataFrame): Testing data targets
        Returns:
            (dict) : best hyper params values found by HyperOpt
        """

        # Get prefect logger
        logger = get_run_logger()

        # Fetch conf
        model_name = conf.mlflow.registered_model_name

        logger.info("Starting training with hyperparameters tuning")

        # Define search space
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(0.001), np.log(10)), # Regularization strength
            'l1_ratio': hp.uniform('l1_ratio', 0.1, 1.0), # L1 vs L2 mix ratio
        }

        # Define hyperopt objective function
        def objective(params):

            with mlflow.start_run():

                # Define the pipeline
                pipeline = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('model', ElasticNet(**params))
                ])

                # Train & Predict
                pipeline.fit(train_x,train_y)
                predicted_y = pipeline.predict(test_x)

                # Evaluate
                (rmse, mae, r2) = evaluation_metrics(test_y, predicted_y)

                # Log params, metrics, model
                mlflow.log_params(params)
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)

                #Infer the model signature
                signature = infer_signature(test_x, predicted_y)
                input_example = test_x.iloc[[0]]

                mlflow.sklearn.log_model(pipeline,
                                         registered_model_name= model_name,
                                         artifact_path= "model",
                                         signature= signature,
                                         input_example= input_example)

            # Set RMSE as loss function :
            return {'loss': rmse, 'status': 'ok'}

        # Minimize Loss function
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=5,
            trials=Trials()
        )

        logger.info(f"Best hyperparameters found: {best_result}")
        logger.info("Training and hyperparameter tuning completed.")

        return best_result



def main():

    # Fetch conf
    conf = read_config()
    train_data_path, test_data_path = conf.data.train_data_path, conf.data.test_data_path
    target_name= conf.data.target_name

    # Prepare databases
    prepare_pipeline_databases(conf,[conf.database.mlflow_db_name,conf.database.monitoring_db_name])

    # Load train test data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    train_x, test_x = train_df.drop([target_name], axis=1), test_df.drop([target_name], axis=1)
    train_y, test_y = train_df[[target_name]], test_df[[target_name]]

    # Training
    set_mlflow_environment(conf)
    train_hyperparameters_tuning(conf,train_x,test_x,train_y,test_y)

if __name__=="__main__":
    main()
