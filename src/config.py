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
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    target_name: str
    test_size: float
    random_state: int

    def __post_init__(self):
        """Resolve all paths relative to base_dir."""
        self.base_dir = Path(__file__).resolve().parents[1]
        self.raw_data_path = self.base_dir / Path(self.raw_data_path)
        self.train_data_path = self.base_dir / Path(self.train_data_path)
        self.test_data_path = self.base_dir / Path(self.test_data_path)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    dialect: str = "postgresql"
    monitoring_db_name: str = "monitoring"
    mlflow_db_name: str = "mlflow"

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


@dataclass
class LoggingConfig:
    level: str


@dataclass
class AppConfig:
    database: DatabaseConfig
    logging: LoggingConfig
    mlflow: MlflowConfig
    data: DataConfig


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
        data=DataConfig(**cfg.data)
    )
    return app_config


if __name__ == "__main__":
    conf = read_config()
    #print(config.mlflow.registered_model_name)
    print(os.getenv("ENV_MODE", "dev"))
    print(conf.database.host)
