import os

import pandas as pd
from prefect import task
from prefect.logging import get_run_logger
from sklearn.model_selection import train_test_split

from config import AppConfig, read_config


def transform_data(data_df):
    # remove spaces from column names
    data_df.columns = data_df.columns.str.replace(" ", "_")
    return data_df


@task(name="data preprocessing & split", retries=3, retry_delay_seconds=5)
def pre_processing(conf:AppConfig,raw_data_df):
    # Get prefect logger
    logger = get_run_logger()

    try:
        # Fetch config
        train_data_path, test_data_path= conf.data.train_data_path, conf.data.test_data_path
        split_test_size, split_random_state= conf.data.test_size, conf.data.random_state
        target_name= conf.data.target_name


        # Transform data
        data_processed_df= transform_data(raw_data_df)

        # Split data
        logger.info(f"Splitting data with test_size={split_test_size}")
        train_df, test_df= train_test_split(data_processed_df, test_size=split_test_size, random_state=split_random_state)
        train_x, test_x = train_df.drop([target_name], axis=1), test_df.drop([target_name], axis=1)
        train_y, test_y = train_df[[target_name]], test_df[[target_name]]

        # Save processed data
        logger.info(f"Saving processed data to {train_data_path} and {test_data_path}")
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        train_df.to_csv(train_data_path,index=False)
        test_df.to_csv(test_data_path,index=False)

        logger.info("Data preprocessing completed.")

        return train_x,test_x,train_y,test_y

    except IOError as e:
        logger.exception(f"Error during file operation during preprocessing: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during preprocessing: {e}")



def main():

    app_conf = read_config()
    raw_data_df = pd.read_csv(app_conf.data.raw_data_path)
    train_x,test_x,train_y,test_y=pre_processing(app_conf, raw_data_df)
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

if __name__ == "__main__":
    main()
