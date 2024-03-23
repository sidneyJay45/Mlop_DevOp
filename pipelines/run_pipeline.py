from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    from zenml.client import Client
     print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
     train_pipeline(data_path="data/olist_customers_dataset.csv")
