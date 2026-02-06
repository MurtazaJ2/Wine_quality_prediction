# model_trainer.py
import os
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from mlflow.models import infer_signature
import xgboost as xgb
import lightgbm as lgb
import importlib
from pathlib import Path
from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience.utils.common import save_json
from mlflow.tracking import MlflowClient


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.models_config = config.models_config
        self.experiment_name = "Wine Quality Multi-Model Comparison"
        self.best_models = {}
        self.overall_best_model = None
        self.overall_best_metrics = None
        self.overall_best_model_name = None
        
    def eval_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        }
    
    def get_model_class(self, model_class_path):
        """Dynamically import model class"""
        module_name, class_name = model_class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    def train_single_model(self, model_name, model_config, X_train, y_train, X_test, y_test):
        """Train and tune a single model"""
        logger.info(f"Training {model_name}...")
        
        best_score = -np.inf
        best_model = None
        best_params = None
        best_metrics = None
        
        # Get model class
        model_class = self.get_model_class(model_config['class'])
        
        # Create parameter grid
        param_grid = model_config['hyperparameters']
        
        # Convert lists to GridSearchCV format
        gs_param_grid = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                gs_param_grid[param] = values
            else:
                gs_param_grid[param] = [values]
        
        # Perform Grid Search
        logger.info(f"Performing GridSearchCV for {model_name}")
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=gs_param_grid,
            scoring='r2',
            cv=self.config.cv_folds,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        metrics = self.eval_metrics(y_test, y_pred)
        
        logger.info(f"{model_name} - Best CV Score: {best_cv_score:.4f}")
        logger.info(f"{model_name} - Test R2: {metrics['r2']:.4f}")
        logger.info(f"{model_name} - Best Params: {best_params}")
        
        return best_model, best_params, metrics
    
    def log_model_to_mlflow(self, model, model_name, params, metrics, X_train, X_test, y_test):
        """Log a single model to MLflow"""
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(f"{model_name}_{param_name}", param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            
            # Log model with appropriate flavor
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Determine model type for proper logging
            model_type = type(model).__name__
            
            # Add tag for model type
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("model_class", model_type)
            mlflow.set_tag("training_time", datetime.now().isoformat())
            
            # Log based on model type
            if model_type == 'XGBRegressor':
                mlflow.xgboost.log_model(model, "model", signature=signature)
            elif model_type == 'LGBMRegressor':
                mlflow.lightgbm.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Log input example
            mlflow.log_input(
                mlflow.data.from_pandas(X_test.iloc[:10]),
                context="test_data_sample"
            )
            
            return run.info.run_id
    
    def train(self):
        """Main training method for all models"""
        # Load data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Starting multi-model training with {len(self.models_config)} models")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Train each model
        for model_name, model_config in self.models_config.items():
            try:
                # Train and tune the model
                model, params, metrics = self.train_single_model(
                    model_name, model_config, X_train, y_train, X_test, y_test
                )
                
                # Log to MLflow
                run_id = self.log_model_to_mlflow(
                    model, model_name, params, metrics, X_train, X_test, y_test
                )
                
                # Store model info
                self.best_models[model_name] = {
                    'model': model,
                    'params': params,
                    'metrics': metrics,
                    'run_id': run_id,
                    'model_type': type(model).__name__
                }
                
                # Update overall best model
                if self.overall_best_model is None or metrics['r2'] > self.overall_best_metrics['r2']:
                    self.overall_best_model = model
                    self.overall_best_metrics = metrics
                    self.overall_best_model_name = model_name
                    self.overall_best_run_id = run_id
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Save all models locally
        self.save_models(X_train, X_test, y_test)
        
        # Create model comparison report
        # self.create_model_comparison_report()
        
        logger.info(f"Training completed. Overall best model: {self.overall_best_model_name}")
        logger.info(f"Best R2: {self.overall_best_metrics['r2']:.4f}")
    
    def save_models(self, X_train, X_test, y_test):
        """Save all trained models locally"""
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        # Save each model
        for model_name, model_info in self.best_models.items():
            model_path = os.path.join(self.config.root_dir, f"{model_name}_model.pkl")
            joblib.dump(model_info['model'], model_path)
            
            # Save model metadata
            metadata = {
                'model_name': model_name,
                'model_type': model_info['model_type'],
                'parameters': model_info['params'],
                'metrics': model_info['metrics'],
                'run_id': model_info['run_id'],
                'feature_names': list(X_train.columns),
                'target_column': self.config.target_column,
                'training_date': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.config.root_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        # Save overall best model with the main name
        best_model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(self.overall_best_model, best_model_path)
        
        # Save best model info
        best_model_info = {
            'best_model_name': self.overall_best_model_name,
            'best_model_type': type(self.overall_best_model).__name__,
            'best_model_metrics': self.overall_best_metrics,
            'best_model_run_id': self.overall_best_run_id,
            'all_models_performance': {
                name: info['metrics'] for name, info in self.best_models.items()
            }
        }
        
        best_info_path = Path(self.config.root_dir, "best_model_info.json")
        save_json(best_info_path, best_model_info)
    
    # def create_model_comparison_report(self):
    #     """Create a comparison report of all models"""
    #     comparison_data = []
        
    #     for model_name, model_info in self.best_models.items():
    #         row = {
    #             'Model': model_name,
    #             'Model_Type': model_info['model_type'],
    #             'R2': model_info['metrics']['r2'],
    #             'RMSE': model_info['metrics']['rmse'],
    #             'MAE': model_info['metrics']['mae'],
    #             'MAPE': model_info['metrics']['mape'],
    #             'Run_ID': model_info['run_id']
    #         }
    #         comparison_data.append(row)
        
    #     df_comparison = pd.DataFrame(comparison_data)
    #     df_comparison = df_comparison.sort_values('R2', ascending=False)
        
    #     # Save comparison report
    #     report_path = os.path.join(self.config.root_dir, "model_comparison.csv")
    #     df_comparison.to_csv(report_path, index=False)
        
    #     # Log comparison to MLflow
    #     with mlflow.start_run(run_name="Model_Comparison_Summary") as run:
    #         mlflow.log_artifact(report_path, "model_comparison")
            
    #         # Log best model info
    #         mlflow.set_tag("overall_best_model", self.overall_best_model_name)
    #         mlflow.log_metric("best_model_r2", self.overall_best_metrics['r2'])
    #         mlflow.log_metric("best_model_rmse", self.overall_best_metrics['rmse'])
            
    #         # Create and log a comparison plot
    #         import matplotlib.pyplot as plt
            
    #         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
    #         # R2 Comparison
    #         models = df_comparison['Model']
    #         r2_scores = df_comparison['R2']
    #         axes[0].bar(models, r2_scores)
    #         axes[0].set_title('Model R2 Scores Comparison')
    #         axes[0].set_xlabel('Model')
    #         axes[0].set_ylabel('R2 Score')
    #         axes[0].tick_params(axis='x', rotation=45)
            
    #         # RMSE Comparison
    #         rmse_scores = df_comparison['RMSE']
    #         axes[1].bar(models, rmse_scores)
    #         axes[1].set_title('Model RMSE Comparison')
    #         axes[1].set_xlabel('Model')
    #         axes[1].set_ylabel('RMSE')
    #         axes[1].tick_params(axis='x', rotation=45)
            
    #         plt.tight_layout()
            
    #         # Save and log plot
    #         plot_path = os.path.join(self.config.root_dir, "model_comparison_plot.png")
    #         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #         mlflow.log_artifact(plot_path, "model_comparison")
    #         plt.close()