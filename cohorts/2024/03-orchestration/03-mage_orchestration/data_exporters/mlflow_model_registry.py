import mlflow
import mlflow.sklearn
from os import path
import pickle
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("linear_regression_yellow_cab")
mlflow.sklearn.autolog()


@data_exporter
def export_model_and_vectorizer_to_mlflow(output):
    dv = output[0]
    lr = output[1]
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        
        # Serialize and log the DictVectorizer
        vectorizer_path = "dict_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(dv, f)
        
        # Log the DictVectorizer as an artifact
        mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")
        
        print("Model and DictVectorizer logged to MLflow")

    os.remove(vectorizer_path)
