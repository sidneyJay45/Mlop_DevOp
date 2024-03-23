import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from SRC.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from sklearn.impute import SimpleImputer
import mlflow 
from zenml.client import Client
from SRC.evaluation import MSE, R2, RMSE

# Define experiment_tracker here
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=str(experiment_tracker.name))
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on the ingested data.

    Args:
    X_train: pd.DataFrame - DataFrame containing the training features.
    X_test: pd.DataFrame - DataFrame containing the test features.
    y_train: pd.DataFrame - DataFrame containing the training labels.
    y_test: pd.DataFrame - DataFrame containing the test labels.
    config: ModelNameConfig - Object containing model name information.

    Returns:
    trained_model: RegressorMixin - Trained regression model.
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model '{config.model_name}' not supported.")
    except Exception as e:
        logging.error(f"Error occurred while training model: {e}")
        raise e
