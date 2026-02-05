import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        param_grid = {
            "alpha": self.config.alpha,
            "l1_ratio": self.config.l1_ratio
        }

        best_r2 = -1
        best_model = None

        mlflow.set_experiment("Wine Quality using ElasticNet")

        for params in ParameterGrid(param_grid):
            with mlflow.start_run():
                logger.info(f"Training with params: {params}")

                model = ElasticNet(
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                    random_state=42
                )
                model.fit(train_x, train_y)

                preds = model.predict(test_x)
                rmse, mae, r2 = self.eval_metrics(test_y, preds)

                mlflow.log_param("alpha", params["alpha"])
                mlflow.log_param("l1_ratio", params["l1_ratio"])
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                signature = infer_signature(train_x, model.predict(train_x))

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=train_x.iloc[:5]
                )

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model

        joblib.dump(
            best_model,
            os.path.join(self.config.root_dir, self.config.model_name)
        )

        logger.info(f"Best model saved with R2: {best_r2}")
