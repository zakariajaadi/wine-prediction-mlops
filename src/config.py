import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load .env file only in dev
if os.getenv("ENV_MODE", "dev") == "dev":
    load_dotenv()

# Data classes to define yaml Conf files
@dataclass
class DataConfig:
    bucket: str
    raw_data_key: str
    train_data_key: str
    test_data_key: str
    target_name: str
    test_size: float
    random_state: int


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    dialect: str = "postgresql"
    monitoring_db_name: str = "monitoring"
    mlflow_db_name: str = "mlflowdb"
    prefect_db_name: str = "prefectdb"


    def construct_db_uri(self, db_name: str):
        """
        Returns a postgres uri with the db name
        """
        return (
            f"{self.dialect}://"
            f"{self.user}:{self.password}@"
            f"{self.host}:{self.port}/"
            f"{db_name}"
        )


@dataclass
class MlflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    model_uri: str
    artifact_root: str


@dataclass
class MinioConfig:
    endpoint_url: str
    access_key: str
    secret_key: str


@dataclass
class LoggingConfig:
    level: str


@dataclass
class AppConfig:
    database: DatabaseConfig
    logging: LoggingConfig
    mlflow: MlflowConfig
    data: DataConfig
    minio: MinioConfig


def read_config():
    base_dir = Path(__file__).resolve().parents[1]

    # Environment mode
    config_path = base_dir / "config" / "dev_config.yaml"

    if os.getenv("ENV_MODE", "dev") == "prod":
        config_path = base_dir / "config" / "prod_config.yaml"

    # Load config
    cfg = OmegaConf.load(config_path)

    app_config = AppConfig(
        database=DatabaseConfig(**cfg.database),
        logging=LoggingConfig(**cfg.logging),
        mlflow=MlflowConfig(**cfg.mlflow),
        minio=MinioConfig(**cfg.minio),
        data=DataConfig(**cfg.data)
    )
    return app_config


if __name__ == "__main__":
    conf = read_config()
    #print(config.mlflow.registered_model_name)
    print(os.getenv("ENV_MODE", "dev"))
    print(conf.database.host)
