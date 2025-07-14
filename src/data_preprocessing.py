from prefect import task
from prefect.logging import get_run_logger
from sklearn.model_selection import train_test_split

from config import AppConfig, read_config
from utils import upload_df_to_minio, download_df_from_minio


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
        data_bucket = conf.data.bucket
        train_data_key, test_data_key= conf.data.train_data_key, conf.data.test_data_key
        split_test_size, split_random_state= conf.data.test_size, conf.data.random_state
        target_name= conf.data.target_name

        # Transform data
        data_processed_df= transform_data(raw_data_df)

        # Split data
        logger.info(f"Splitting data with test_size={split_test_size}")
        train_df, test_df= train_test_split(data_processed_df, test_size=split_test_size, random_state=split_random_state)
        train_x, test_x = train_df.drop([target_name], axis=1), test_df.drop([target_name], axis=1)
        train_y, test_y = train_df[[target_name]], test_df[[target_name]]

        # Save processed data to MinIO
        logger.info(f"Uploading processed train data to s3://{data_bucket}/{train_data_key}")
        upload_df_to_minio(train_df, bucket=data_bucket, destination_key=train_data_key)

        logger.info(f"Uploading processed test data to s3://{data_bucket}/{test_data_key}")
        upload_df_to_minio(test_df, bucket=data_bucket, destination_key=test_data_key)

        logger.info("Data preprocessing completed.")

        return train_x, test_x, train_y, test_y

    except IOError as e:
        print(f"Error during file operation during preprocessing: {e}")
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")



def main():

    app_conf = read_config()
    # Download raw data from minio
    raw_data_df = download_df_from_minio(bucket=app_conf.data.bucket, object_key=app_conf.data.raw_data_key)
    # Preprocess data and save to Minio
    train_x, test_x, train_y, test_y = pre_processing(app_conf, raw_data_df)
    # Print processed data shapes
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

if __name__ == "__main__":
    main()
