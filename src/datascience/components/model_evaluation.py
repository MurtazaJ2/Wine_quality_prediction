# model_evaluation.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature, MetricThreshold

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json
from src.datascience import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_uri
        
    def eval_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "explained_variance": explained_variance
        }
    
    def register_best_model(self, model_info_path, test_data):
        """Register the overall best model to MLflow Model Registry"""
        # Load best model info
        with open(model_info_path, 'r') as f:
            best_model_info = json.load(f)
        
        model = joblib.load(self.config.model_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        # Get predictions
        predictions = model.predict(test_x)
        metrics = self.eval_metrics(test_y, predictions)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        logger.info(f"Registering best model: {best_model_info['best_model_name']}")
        logger.info(f"Model metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        if tracking_url_type_store != "file":
            # Use the existing run to register the model
            model_uri = f"runs:/{best_model_info['best_model_run_id']}/model"
            
            try:
                # Register model with versioning
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="WineQuality_Best_Model"
                )
                
                # Add metadata to the registered model
                client = mlflow.tracking.MlflowClient()
                
                # Set model description
                description = f"""
                Best performing model for Wine Quality prediction.
                Model Type: {best_model_info['best_model_type']}
                Performance: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}
                Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                client.update_registered_model(
                    name=registered_model.name,
                    description=description
                )
                
                # Add version description
                client.update_model_version(
                    name=registered_model.name,
                    version=registered_model.version,
                    description=f"Version {registered_model.version}: {best_model_info['best_model_name']}"
                )
                
                # Transition model to Staging
                client.transition_model_version_stage(
                    name=registered_model.name,
                    version=registered_model.version,
                    stage="Staging"
                )
                
                logger.info(f"Model registered successfully: {registered_model.name}")
                logger.info(f"Version: {registered_model.version}")
                logger.info(f"Stage: Staging")
                
                # Save registration info
                registration_info = {
                    'registered_model_name': registered_model.name,
                    'registered_model_version': registered_model.version,
                    'original_run_id': best_model_info['best_model_run_id'],
                    'model_type': best_model_info['best_model_type'],
                    'registration_date': datetime.now().isoformat(),
                    'metrics': metrics
                }
                
                reg_info_path = Path(self.config.root_dir, "model_registration_info.json")
                save_json(reg_info_path, registration_info)
                
                return registered_model
                
            except Exception as e:
                logger.error(f"Error registering model: {str(e)}")
                raise
        else:
            logger.info("Local file store detected. Model registration skipped.")
            return None
    
    def create_model_card(self, model, model_info, metrics, test_data):
        """Create a model card for documentation"""
        test_x = test_data.drop([self.config.target_column], axis=1)
        
        model_card = {
            'model_name': model_info.get('best_model_name', 'Unknown'),
            'model_type': type(model).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'performance_metrics': metrics,
            'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
            'input_features': list(test_x.columns),
            'target_feature': self.config.target_column,
            'model_description': 'Wine Quality Prediction Model',
            'intended_use': 'Predict wine quality based on chemical properties',
            'limitations': 'Model performance may vary with different wine types',
            'ethical_considerations': 'None identified',
            'maintainers': ['Data Science Team']
        }
        
        model_card_path = Path(self.config.root_dir, "model_card.json")
        save_json(Path(model_card_path), model_card)
        
        return model_card
    
    def log_into_mlflow(self):
        """Main method to evaluate and register models"""
        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        # Load best model
        model = joblib.load(self.config.model_path)
        
        # Predict
        predictions = model.predict(test_x)
        metrics = self.eval_metrics(test_y, predictions)
        
        # Save metrics locally
        save_json(Path(self.config.metric_file_name), metrics)
        
        # Create model card
        best_model_info_path = Path(
            os.path.dirname(self.config.model_path),
            "best_model_info.json"
        )
        
        with open(best_model_info_path, 'r') as f:
            best_model_info = json.load(f)
        
        self.create_model_card(model, best_model_info, metrics, test_data)
        
        # Register best model
        registered_model = self.register_best_model(best_model_info_path, test_data)
        
        # if registered_model:
        #     # Log additional validation
        #     self.perform_additional_validation(model, test_data, registered_model)
            
        logger.info("Model evaluation completed successfully.")
        
        return metrics
    
    # def perform_additional_validation(self, model, test_data, registered_model):
    #     """Perform additional validation tests"""
    #     test_x = test_data.drop([self.config.target_column], axis=1)
        
    #     # Feature importance analysis
    #     if hasattr(model, 'feature_importances_'):
    #         feature_importance = pd.DataFrame({
    #             'feature': test_x.columns,
    #             'importance': model.feature_importances_
    #         }).sort_values('importance', ascending=False)
            
    #         importance_path = Path(self.config.root_dir, "feature_importance.csv")
    #         feature_importance.to_csv(importance_path, index=False)
            
    #         # Log to MLflow
    #         with mlflow.start_run(run_name="Feature_Importance_Analysis"):
    #             mlflow.log_artifact(importance_path, "validation")
    #             mlflow.set_tag("analysis_type", "feature_importance")
        
    #     # Create validation report
    #     validation_report = {
    #         'model_name': registered_model.name,
    #         'model_version': registered_model.version,
    #         'validation_date': datetime.now().isoformat(),
    #         'validation_tests': [
    #             'basic_metrics_calculation',
    #             'model_registration',
    #             'feature_importance_analysis'
    #         ],
    #         'status': 'PASSED'
    #     }
        
    #     validation_path = Path(self.config.root_dir, "validation_report.json")
    #     save_json(validation_path, validation_report)