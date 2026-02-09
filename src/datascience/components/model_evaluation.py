# model_evaluation.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json
from src.datascience import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_uri
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_registry_uri(config.mlflow_uri)

    def eval_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(
            np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))
        ) * 100

        explained_variance = (
            1 - np.var(y_true - y_pred) / np.var(y_true)
            if np.var(y_true) > 0 else 0
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "explained_variance": explained_variance
        }

    def register_best_model(self, best_model_info_path: Path):
        """
        Register ONLY the best model across all algorithms
        using best_model_run_id → runs:/<run_id>/model
        """
        with open(best_model_info_path, "r") as f:
            best_model_info = json.load(f)

        best_run_id = best_model_info["best_model_run_id"]
        model_uri = f"runs:/{best_run_id}/model"

        logger.info("Registering overall best model")
        logger.info(f"Best algorithm: {best_model_info['best_model_name']}")
        logger.info(f"Run ID: {best_run_id}")

        tracking_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_store == "file":
            logger.info("File-based MLflow store detected. Skipping registry.")
            return None

        client = mlflow.tracking.MlflowClient()

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="WineQuality_Best_Model"
        )

        # Add descriptions
        client.update_registered_model(
            name=registered_model.name,
            description=(
                "Champion model selected across all algorithms "
                "based on highest test R2 score."
            )
        )

        client.update_model_version(
            name=registered_model.name,
            version=registered_model.version,
            description=(
                f"Selected model: {best_model_info['best_model_name']} | "
                f"R2: {best_model_info['best_model_metrics']['r2']:.4f}"
            )
        )

        # Move to Staging
        client.transition_model_version_stage(
            name=registered_model.name,
            version=registered_model.version,
            stage="Staging"
        )

        registration_info = {
            "registered_model_name": registered_model.name,
            "registered_model_version": registered_model.version,
            "source_run_id": best_run_id,
            "model_type": best_model_info["best_model_type"],
            "registration_date": datetime.now().isoformat(),
            "selection_metric": "test_r2",
            "metrics": best_model_info["best_model_metrics"]
        }

        save_json(
            Path(self.config.root_dir, "model_registration_info.json"),
            registration_info
        )

        logger.info(
            f"Registered {registered_model.name} v{registered_model.version} "
            f"to Staging"
        )

        return registered_model

    def log_into_mlflow(self):
        """
        Evaluation entry point.
        Assumes training has already produced best_model_info.json
        """
        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Load locally saved best model (for metrics + model card only)
        model = joblib.load(self.config.model_path)

        predictions = model.predict(test_x)
        metrics = self.eval_metrics(test_y, predictions)

        save_json(Path(self.config.metric_file_name), metrics)

        best_model_info_path = Path(
            os.path.dirname(self.config.model_path),
            "best_model_info.json"
        )

        # Register ONLY the overall best model
        self.register_best_model(best_model_info_path)

        logger.info("Model evaluation & registration completed successfully")
        return metrics
