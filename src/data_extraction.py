import pandas as pd
from prefect import task
from prefect.logging import get_run_logger

from config import AppConfig, read_config
from utils import upload_df_to_minio


@task(name="data extraction",retries=3, retry_delay_seconds=5)
def extract_data(conf:AppConfig):

    # Get Prefect Logger
    logger = get_run_logger()

    # Fetch config
    data_bucket=conf.data.bucket
    raw_data_key=conf.data.raw_data_key

    # Data url
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        # Extract data
        raw_data_df=pd.read_csv(csv_url,sep=";")

        # Save data to MinIO
        upload_df_to_minio(raw_data_df,data_bucket,raw_data_key)
        logger.info(f"Data extraction completed and saved to MinIO at s3://{data_bucket}/{raw_data_key}")

        return raw_data_df

    except Exception as e:
        logger.exception(f"Data extraction error")
        raise



def main():
            conf = read_config()
            raw_data_df=extract_data(conf)
            print(raw_data_df.head())

if __name__ == "__main__":
    main()


