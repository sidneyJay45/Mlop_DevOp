# step 2 : We will clean our dataset
import logging

import pandas as pd
from zenml import step
from typing import Tuple
from SRC.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series, "y_test"],
]: 
    """
    Clean the data by applying the specified strategy i.e divide into train and test sets
    Args:
        df:Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train,y_test
    except Exception as e:
        logging.error("Error occurred while cleaning data: {}".format(e))
        raise e



import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):

    """
    Strategy for pre-processing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try: #dropping the columns not needed for the model
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )
            #filling the null values with the median
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(),
                inplace=True
            )
            data["product_length_cm"].fillna(
              data["product_length_cm"].median(),
              inplace=True
            )
            data["product_height_cm"].fillna(
              data["product_height_cm"].median(),
              inplace=True
            )
            data["product_width_cm"].fillna(
              data["product_width_cm"].median(),
              inplace=True
            )



            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"] #dropped the columns not needed for the model
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error occurred while pre-processing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing sets.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e


class DataCleaning:
    """
    Class for cleaning the data which processes the data and divides it into training and test sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data by applying the specified strategy.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error occurred while handling data: {}".format(e))
            raise e





import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):

    """
    Strategy for pre-processing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try: #dropping the columns not needed for the model
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )
            #filling the null values with the median
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(),
                inplace=True
            )
            data["product_length_cm"].fillna(
              data["product_length_cm"].median(),
              inplace=True
            )
            data["product_height_cm"].fillna(
              data["product_height_cm"].median(),
              inplace=True
            )
            data["product_width_cm"].fillna(
              data["product_width_cm"].median(),
              inplace=True
            )



            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"] #dropped the columns not needed for the model
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error occurred while pre-processing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing sets.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e


class DataCleaning:
    """
    Class for cleaning the data which processes the data and divides it into training and test sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data by applying the specified strategy.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error occurred while handling data: {}".format(e))
            raise e




experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"],
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



experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"],
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



import logging
import pandas as pd
from zenml import step
from SRC.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from sklearn.impute import SimpleImputer
import mlflow 
from zenml.client import Client
from typing import Tuple
from typing_extensions import Annotated


# Define experiment_tracker here
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"],
]:
    """
    Evaluates the performance of the model using the provided DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the data to be used for evaluation.

    Returns:
    None
    """
    experiment_tracker = Client().active_stack.experiment_tracker
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





# Define experiment_tracker here
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"],
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


#model_train.py
import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from SRC.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow 
from zenml.client import Client
from SRC.evaluation import MSE, R2, RMSE

from sklearn.impute import SimpleImputer

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"],
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


train_pipeline(data_path="data/olist_customers_dataset.csv")



mlflow ui --backend-store-uri 'file:/home/runner/tryit/.config/zenml/local_stores/7e5943dc-4bdb-402b-8e93-2b3df583bcc5/mlruns'



@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}"
            f" step {pipeline_step_name} and model {model_name}"
            f"pipeline for the '{model_name}' model is currently running."
        )
    return existing_services[0]



import numpy as np
import json
from typing import cast
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings, docker_settings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from .utils import get_data_for_test
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

# The criteria for the model in terms of accuracy
class DeploymentTriggerConfig(BaseParameters):
    """ DeploymentTriggerConfig"""
    min_accuracy: float = 0

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    if data is None:
        raise ValueError("Data is None, unable to return a valid string")
    return str(data)


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implement a simple deployment trigger that looks at the input model accuracy and decides if it's good enough to deploy or not"""
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True 

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}"
            f" step {pipeline_step_name} and model {model_name}"
            f"pipeline for the '{model_name}' model is currently running."
        )
    return cast(MLFlowDeploymentService, existing_services[0])


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) ->np.ndarray:
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)

    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction


#initial page ya streamlit_app.py
import json
import numpy as np
import pandas as pd
import streamlit as st


from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run

# Define your Streamlit app code below
def run():
    st.markdown(
        """ 
        #### Problem Statement 
        The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.
        """
    )

    st.markdown(
        """ 
        #### Description of Features 
        This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score.
        | Models | Description |
        | ------------- | - |
        | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. |
        | Payment Installments | Number of installments chosen by the customer. |
        | Payment Value | Total amount paid by the customer. |
        | Price | Price of the product. |
        | Freight Value | Freight value of the product. |
        | Product Name length | Length of the product name. |
        | Product Description length | Length of the product description. |
        | Product photos Quantity | Number of product published photos |
        | Product weight measured in grams | Weight of the product measured in grams. |
        | Product length (CMs) | Length of the product measured in centimeters. |
        | Product height (CMs) | Height of the product measured in centimeters. |
        | Product width (CMs) | Width of the product measured in centimeters. |
        """
    )

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("Freight Value")
    product_name_length = st.number_input("Product Name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run()

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_length": [product_name_length],
                "product_description_length": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfaction rate (range between 0 - 5) with given product details is: {}".format(
                pred
            )
        )

if __name__ == "__main__":
    run()

#original run_deployment.py
from rich import print
from typing import cast
import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model ('deploy'), or to "
    "only run a prediction against the deployed model "
    "('predict'). By default both will be run "
    "('deploy_and_predict')."
)
@click.option(
    "--min-accuracy",
    default=0,
    help="Minimum accuracy required to deploy the model"
)
def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(data_path="data/olist_customers_dataset.csv",
                                       min_accuracy=min_accuracy,
                                       workers=1,
                                       timeout=60,)
    if predict:
        inference_pipeline(
          pipeline_name="continuous_deployment_pipeline",
          pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )

if __name__ == "__main__":
    run_deployment()
#original streamlit_app.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment




def cedo():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    st.markdown(
        """
        #### Problem Statement
        The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.
        """
    )

    st.markdown(
        """
        #### Description of Features
        This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score.
        | Models | Description |
        | ------------- | - |
        | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. |
        | Payment Installments | Number of installments chosen by the customer. |
        | Payment Value | Total amount paid by the customer. |
        | Price | Price of the product. |
        | Freight Value | Freight value of the product. |
        | Product Name length | Length of the product name. |
        | Product Description length | Length of the product description. |
        | Product photos Quantity | Number of product published photos |
        | Product weight measured in grams | Weight of the product measured in grams. |
        | Product length (CMs) | Length of the product measured in centimeters. |
        | Product height (CMs) | Height of the product measured in centimeters. |
        | Product width (CMs) | Width of the product measured in centimeters. |
        """
    )

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("Freight Value")
    product_name_length = st.number_input("Product Name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run()

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_length": [product_name_length],
                "product_description_length": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfaction rate (range between 0 - 5) with given product details is: {}".format(
                pred
            )
        )

    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree-based models and compared the performance of each model. The results are as follows:"
        )

        df_results = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df_results)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes to predicting customer satisfaction rate."
        )
        # Freeze all images
        # image = Image.open("_assets/feature_importance_gain.png")
        # st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    cedo()



#mlflow ui to run to see the model
mlflow ui --backend-store-uri 'file:/home/runner/tryit/.config/zenml/local_stores/7e5943dc-4bdb-402b-8e93-2b3df583bcc5/mlruns'
'file:/home/runner/tryit/.config/zenml/local_stores/7e
5943dc-4bdb-402b-8e93-2b3df583bcc5/mlruns

mlflow ui --backend-store-uri 'file:/home/runner/tryit/.config/zenml/local_stores/7e5943dc-4bdb-402b-8e93-2b3df583bcc5/mlruns'

#mlflow ui --backend-store-uri 'file:/home/runner/tryit/.config/