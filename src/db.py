from typing import List

import psycopg2
from prefect import task
from prefect.logging import get_run_logger

from config import AppConfig


@task(name="Ensure pipeline databases")
def prepare_pipeline_databases(conf: AppConfig, db_names_list: List[str]):
    """
    Ensures or create necessary databases in param
    """
    #Get prefect logger
    logger = get_run_logger()

    # Iterate over db_names_list param to ensure databases
    for db_name in db_names_list:
        try:
            # Fetch conf
            postgres_db_uri = conf.database.construct_db_uri("postgres")

            # Connect to 'postgres' DB to check if db_name exists
            con = psycopg2.connect(postgres_db_uri)
            con.autocommit = True
            cursor = con.cursor()
            cursor.execute(f"SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname='{db_name}')")
            is_exist = cursor.fetchone()[0]

            # Create database
            if not is_exist:
                cursor.execute(f"CREATE DATABASE {db_name}")
                logger.info(f"Database {db_name} created and is ready for use.")

            cursor.close()
            con.close()

        except Exception:
            logger.exception(f"Error preparing database for {db_name}")
            raise

    logger.info(f"All databases are available.")
