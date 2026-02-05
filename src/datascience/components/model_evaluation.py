import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json


os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            save_json(
                path=Path(self.config.metric_file_name),
                data=scores
            )

            signature = infer_signature(test_x, predicted_qualities)

            mlflow.log_param("alpha", model.alpha)
            mlflow.log_param("l1_ratio", model.l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Register model if not using file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=test_x.iloc[:5],
                    registered_model_name="ElasticnetModel"
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model"
                )
