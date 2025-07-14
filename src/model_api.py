import logging
import os

import pandas as pd
from fastapi import FastAPI

from src.utils import InputDataModel
from src.utils import make_prediction_from_model, save_predictions

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Filter for healthcheck
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

# FastAPI app
app = FastAPI()


# HealthCheck Endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Prediction Endpoint
@app.post("/predict")
async def predict(data: InputDataModel):
    """
    endpoint for model serving

    Args:
      data (InputDataModel) : a pydantic for inference data
    Returns:
      (json) : predictions of wine quality
               Example: {'prediction': [5,6 ...]}
    """

    # Serialize pydantic
    data_dict = data.model_dump()  # {'inputs':[{'feature1':val1,'feature2': val2 ...}, ... ]}

    # Create a dataframe with serialized data (no need to specify column names)
    samples_data_df = pd.DataFrame(data=data_dict['inputs'])

    # Get model uri
    model_uri = os.getenv('MODEL_URI')

    # Make predictions
    predictions_list = make_prediction_from_model(samples_data_df, model_uri)

    # Save predictions
    save_predictions(samples_data_df, predictions_list)

    return {'predictions': predictions_list}





