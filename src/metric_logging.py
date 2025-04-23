import os
import mlflow
import mlflow.sklearn

# Set up MLflow tracking URI and authentication for DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/vamsisaigarapati/bitcoin_price_pred_CSE574.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'vamsisaigarapati'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0d66986d30f48a915d60b73c435bdae6ee103eb8'

# Configure MLflow
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment("Bitcoin_Price_Prediction_CSE574")

def log_experiment(model_name, params, metrics, model_object=None):
    """
    Logs model, parameters, and evaluation metrics to MLflow (DagsHub).
    Model logging is optional.
    
    :param model_name: (str) Name of the model (e.g., "ARIMA", "XGBoost", "LSTM").
    :param params: (dict) Hyperparameters used for the model.
    :param metrics: (dict) Performance metrics (e.g., RMSE, MAPE).
    :param model_object: (optional) Trained model object to be saved (e.g., sklearn, SARIMA, etc.)
    """
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # 1. Save the model only if provided
        if model_object is not None:
            mlflow.sklearn.log_model(
                sk_model=model_object,
                artifact_path="model",
                registered_model_name=f"{model_name}_Model"
            )

        # 2. Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # 3. Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # 4. Set a tag for better tracking
        mlflow.set_tag("mlflow.runName", f"{model_name}")

        print(f"✅ {model_name} logged successfully to DagsHub MLflow.")
