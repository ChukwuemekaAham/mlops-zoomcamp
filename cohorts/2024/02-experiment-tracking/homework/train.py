import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("mlops_zoomcamp")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    with mlflow.start_run():
        
        mlflow.set_tag("developer","melanie")
        mlflow.set_tag("model","random_forest_regressor")
        mlflow.log_param("train_data_path","./data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid_data_path","./data/green_tripdata_2023-02.parquet")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))


        rf = RandomForestRegressor(max_depth=10, random_state=0)
        mlflow.log_param("max_depth","10")
        mlflow.log_param("random_state","0")
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric(key="rmse",value=rmse)
        mlflow.end_run()


if __name__ == '__main__':
    run_train()