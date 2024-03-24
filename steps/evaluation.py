import logging
from typing import Tuple
from typing_extensions import Annotated

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from zenml.client import Client

from SRC.evaluation import MSE, R2, RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"]
]:
    """
    Evaluates the performance of the model using the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be used for evaluation.

    Returns:
        None
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error occurred while evaluating model: {}".format(e))
        raise e
