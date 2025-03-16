import os
import mlflow

# Set up MLflow tracking URI and authentication for DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/vamsisaigarapati/bitcoin_price_pred_CSE574.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'vamsisaigarapati'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0d66986d30f48a915d60b73c435bdae6ee103eb8'

# Configure MLflow
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment("Bitcoin_Price_Prediction_CSE574")

def log_experiment(model_name, params, metrics):
    """
    Logs model parameters and evaluation metrics to MLflow (DagsHub).
    Dynamically adds new parameters or metrics if they were not previously logged.
    
    :param model_name: (str) Name of the model (e.g., "ARIMA", "XGBoost", "LSTM").
    :param params: (dict) Hyperparameters used for the model.
    :param metrics: (dict) Performance metrics (e.g., RMSE, MAPE).
    """

    # Start a new MLflow run
    with mlflow.start_run():
        # Fetch previous runs to check existing parameters
        previous_runs = mlflow.search_runs(order_by=["start_time DESC"])
        
        # Check if there were any previous runs for this model
        if not previous_runs.empty:
            last_run_id = previous_runs.iloc[0]["run_id"]
            last_run_data = mlflow.get_run(last_run_id).data

            # Retrieve existing parameters and metrics
            existing_params = last_run_data.params
            existing_metrics = last_run_data.metrics
        else:
            existing_params = {}
            existing_metrics = {}

        # Log only new parameters (skip already logged ones)
        for param_name, param_value in params.items():
            if param_name not in existing_params:
                mlflow.log_param(param_name, param_value)

        # Set experiment run name dynamically based on model
        mlflow.set_tag("mlflow.runName", f"{model_name}-Baseline")

        # Log new metrics (skip already logged ones)
        for metric_name, metric_value in metrics.items():
            if metric_name not in existing_metrics:
                mlflow.log_metric(metric_name, metric_value)
        print('jai_balayya')

        print(f"✅ {model_name} metrics logged successfully to DagsHub MLflow.")

# def log_experiment(model_name, params, metrics):
#     """
#     Logs model parameters and evaluation metrics to MLflow (DagsHub).
    
#     :param model_name: (str) Name of the model (e.g., "ARIMA", "XGBoost", "LSTM").
#     :param params: (dict) Hyperparameters used for the model.
#     :param metrics: (dict) Performance metrics (e.g., RMSE, MAPE).
#     """
#     with mlflow.start_run():
#         # Log hyperparameters
#         mlflow.log_params(params)
        
#         # Set experiment run name dynamically based on model
#         mlflow.set_tag("mlflow.runName", f"{model_name}-Baseline")

#         # Log evaluation metrics
#         for metric_name, metric_value in metrics.items():
#             mlflow.log_metric(metric_name, metric_value)

#         print(f"✅ {model_name} metrics logged successfully to DagsHub MLflow.")