import pandas as pd
from pipelines.training_pipeline import train_pipeline
from zenml.core.client import Client

import subprocess

# Define your custom date parser function
def date_parser(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')  # Adjust the format according to your date format
    except ValueError:
        return pd.NaT  # Return NaT (Not a Time) for non-date values

# Read the CSV file with the custom date parser

if __name__ == "__main__":
    # Run the pipelinezen
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")

    # Launch MLflow UI with backend store URI
mlflow ui --backend-store-uri 'file:/home/runner/tryit/.config/zenml/local_stores/7e5943dc-4bdb-402b-8e93-2b3df583bcc5/mlruns'
correct code

