import os


import pandas as pd
from prefect import task
from prefect.logging import get_run_logger

from config import AppConfig, read_config


@task(name="data extraction",retries=3, retry_delay_seconds=5)
def extract_data(conf:AppConfig):

    # Get Prefect Logger
    logger = get_run_logger()

    # Fetch config
    raw_data_path=conf.data.raw_data_path

    # Data url
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        # Extract data
        raw_data_df=pd.read_csv(csv_url,sep=";")

        # Save data to csv
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        raw_data_df.to_csv(raw_data_path, sep=",", index=False)

        logger.info(f"Data extraction completed. Saving extracted data to {raw_data_path}")

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


