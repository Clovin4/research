import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.create_experiment("mean_risk")

mlflow.set_experiment("check-localhost-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 6)
    mlflow.log_metric("bar", 2)
