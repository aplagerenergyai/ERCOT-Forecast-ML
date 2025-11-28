"""
train_quantile.py

Train Quantile Regression model to predict DART spread with uncertainty estimates.

Predicts three quantiles:
  - P10 (10th percentile): Lower bound
  - P50 (50th percentile): Median prediction
  - P90 (90th percentile): Upper bound

This provides an 80% prediction interval (P10-P90).
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model, evaluate_quantile_model, log_metrics_to_mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantileDARTModel(BaseEstimator, RegressorMixin):
    """
    Quantile regression model for DART spread with uncertainty quantification.
    
    Trains three separate LightGBM models to predict different quantiles:
      - P10 (alpha=0.1): 10th percentile (lower bound)
      - P50 (alpha=0.5): 50th percentile (median)
      - P90 (alpha=0.9): 90th percentile (upper bound)
    
    This provides:
      1. Point prediction (P50/median)
      2. Prediction interval (P10-P90) for uncertainty quantification
      3. Risk assessment (how wide is the interval?)
    
    Implements standard scikit-learn interface:
    - .fit(X, y) for training
    - .predict(X) returns dictionary with p10, p50, p90 predictions
    """
    
    def __init__(
        self,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=96,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        verbose=-1,
        device='gpu'
    ):
        """Initialize the quantile regression model with LightGBM hyperparameters."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.verbose = verbose
        self.device = device
        
        # Three models will be trained in .fit()
        self.model_p10 = None
        self.model_p50 = None
        self.model_p90 = None
    
    def _get_lgb_params(self, alpha: float):
        """
        Get LightGBM parameters for quantile regression.
        
        Args:
            alpha: Quantile level (0.1, 0.5, or 0.9)
        """
        params = {
            'objective': 'quantile',
            'alpha': alpha,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 5,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'verbose': self.verbose,
        }
        
        # Add GPU params if device is GPU
        if self.device == 'gpu':
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        return params
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train all three quantile models.
        
        Args:
            X: Feature matrix (training)
            y: Target DART spread (training)
            X_val: Validation features (optional)
            y_val: Validation DART spread (optional)
        """
        logger.info("="*80)
        logger.info("QUANTILE REGRESSION TRAINING")
        logger.info("="*80)
        logger.info("Training 3 quantile models: P10, P50, P90")
        
        # Train P10 model
        logger.info("\n" + "="*80)
        logger.info("TRAINING P10 MODEL (10th Percentile - Lower Bound)")
        logger.info("="*80)
        
        params_p10 = self._get_lgb_params(alpha=0.1)
        train_data_p10 = lgb.Dataset(X, label=y)
        
        callbacks_p10 = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets_p10 = [train_data_p10]
        valid_names_p10 = ['train']
        
        if X_val is not None and y_val is not None:
            val_data_p10 = lgb.Dataset(X_val, label=y_val, reference=train_data_p10)
            valid_sets_p10.append(val_data_p10)
            valid_names_p10.append('val')
        
        self.model_p10 = lgb.train(
            params_p10,
            train_data_p10,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_p10,
            valid_names=valid_names_p10,
            callbacks=callbacks_p10
        )
        
        logger.info(f"✓ P10 model trained. Best iteration: {self.model_p10.best_iteration}")
        
        # Train P50 model
        logger.info("\n" + "="*80)
        logger.info("TRAINING P50 MODEL (50th Percentile - Median)")
        logger.info("="*80)
        
        params_p50 = self._get_lgb_params(alpha=0.5)
        train_data_p50 = lgb.Dataset(X, label=y)
        
        callbacks_p50 = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets_p50 = [train_data_p50]
        valid_names_p50 = ['train']
        
        if X_val is not None and y_val is not None:
            val_data_p50 = lgb.Dataset(X_val, label=y_val, reference=train_data_p50)
            valid_sets_p50.append(val_data_p50)
            valid_names_p50.append('val')
        
        self.model_p50 = lgb.train(
            params_p50,
            train_data_p50,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_p50,
            valid_names=valid_names_p50,
            callbacks=callbacks_p50
        )
        
        logger.info(f"✓ P50 model trained. Best iteration: {self.model_p50.best_iteration}")
        
        # Train P90 model
        logger.info("\n" + "="*80)
        logger.info("TRAINING P90 MODEL (90th Percentile - Upper Bound)")
        logger.info("="*80)
        
        params_p90 = self._get_lgb_params(alpha=0.9)
        train_data_p90 = lgb.Dataset(X, label=y)
        
        callbacks_p90 = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets_p90 = [train_data_p90]
        valid_names_p90 = ['train']
        
        if X_val is not None and y_val is not None:
            val_data_p90 = lgb.Dataset(X_val, label=y_val, reference=train_data_p90)
            valid_sets_p90.append(val_data_p90)
            valid_names_p90.append('val')
        
        self.model_p90 = lgb.train(
            params_p90,
            train_data_p90,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_p90,
            valid_names=valid_names_p90,
            callbacks=callbacks_p90
        )
        
        logger.info(f"✓ P90 model trained. Best iteration: {self.model_p90.best_iteration}")
        
        logger.info("\n" + "="*80)
        logger.info("QUANTILE REGRESSION TRAINING COMPLETE")
        logger.info("="*80)
        
        return self
    
    def predict(self, X) -> Dict[str, np.ndarray]:
        """
        Predict all three quantiles.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary with keys 'p10', 'p50', 'p90' containing predictions
        """
        if self.model_p10 is None or self.model_p50 is None or self.model_p90 is None:
            raise RuntimeError("Models must be trained before prediction. Call .fit() first.")
        
        # Predict each quantile
        p10 = self.model_p10.predict(X, num_iteration=self.model_p10.best_iteration)
        p50 = self.model_p50.predict(X, num_iteration=self.model_p50.best_iteration)
        p90 = self.model_p90.predict(X, num_iteration=self.model_p90.best_iteration)
        
        return {
            'p10': p10,
            'p50': p50,
            'p90': p90
        }
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance from all three models."""
        if self.model_p10 is None or self.model_p50 is None or self.model_p90 is None:
            raise RuntimeError("Models must be trained before getting feature importance.")
        
        p10_importance = self.model_p10.feature_importance(importance_type=importance_type)
        p50_importance = self.model_p50.feature_importance(importance_type=importance_type)
        p90_importance = self.model_p90.feature_importance(importance_type=importance_type)
        
        # Average importance across all three models
        avg_importance = (p10_importance + p50_importance + p90_importance) / 3
        
        return {
            'p10': p10_importance,
            'p50': p50_importance,
            'p90': p90_importance,
            'average': avg_importance
        }


def main():
    """Main training pipeline."""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Train Quantile Regression model for DART spread')
        parser.add_argument('--features_path', type=str, default=None, 
                            help='Path to features directory or parquet file')
        args = parser.parse_args()
        
        logger.info("="*80)
        logger.info("QUANTILE DART TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        if args.features_path:
            features_path = os.path.join(args.features_path, "hourly_features.parquet")
            logger.info(f"Using command-line features path: {features_path}")
        else:
            features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Initialize and train model
        model = QuantileDARTModel(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=96,
            min_child_samples=60,
            subsample=0.8,
            colsample_bytree=0.8,
            device='gpu'
        )
        
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val
        )
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train)
        train_metrics = evaluate_quantile_model(y_train, y_train_pred, "Train")
        log_metrics_to_mlflow({
            'mae_p50': train_metrics['mae_p50'],
            'rmse_p50': train_metrics['rmse_p50'],
            'r2_p50': train_metrics['r2_p50'],
            'coverage_80': train_metrics['coverage_80'],
            'interval_mean_width': train_metrics['interval_mean_width']
        }, "train")
        
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_quantile_model(y_val, y_val_pred, "Validation")
        log_metrics_to_mlflow({
            'mae_p50': val_metrics['mae_p50'],
            'rmse_p50': val_metrics['rmse_p50'],
            'r2_p50': val_metrics['r2_p50'],
            'coverage_80': val_metrics['coverage_80'],
            'interval_mean_width': val_metrics['interval_mean_width']
        }, "val")
        
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_quantile_model(y_test, y_test_pred, "Test")
        log_metrics_to_mlflow({
            'mae_p50': test_metrics['mae_p50'],
            'rmse_p50': test_metrics['rmse_p50'],
            'r2_p50': test_metrics['r2_p50'],
            'coverage_80': test_metrics['coverage_80'],
            'interval_mean_width': test_metrics['interval_mean_width']
        }, "test")
        
        # Sample predictions
        logger.info("\n" + "="*80)
        logger.info("SAMPLE PREDICTIONS WITH UNCERTAINTY")
        logger.info("="*80)
        
        logger.info("\nFirst 10 test predictions:")
        logger.info(f"{'Actual':<12} {'P10':<12} {'P50':<12} {'P90':<12} {'Width':<12} {'In Interval':<12}")
        logger.info("-" * 80)
        for i in range(min(10, len(y_test))):
            actual = y_test[i]
            p10 = y_test_pred['p10'][i]
            p50 = y_test_pred['p50'][i]
            p90 = y_test_pred['p90'][i]
            width = p90 - p10
            in_interval = "✓" if p10 <= actual <= p90 else "✗"
            
            logger.info(
                f"${actual:>10.2f}  "
                f"${p10:>10.2f}  "
                f"${p50:>10.2f}  "
                f"${p90:>10.2f}  "
                f"${width:>10.2f}  "
                f"{in_interval:>10s}"
            )
        
        # Feature importance
        logger.info("\n" + "="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES (AVERAGED ACROSS QUANTILES)")
        logger.info("="*80)
        
        importance_dict = model.get_feature_importance(importance_type='gain')
        feature_names = loader.feature_columns
        
        importance_df = sorted(
            zip(feature_names, importance_dict['average']),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        for i, (name, importance) in enumerate(importance_df, 1):
            logger.info(f"  {i:2d}. {name:50s} {importance:10.2f}")
        
        # Save model
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        # Get output path from Azure ML or use local
        output_path = os.environ.get("AZUREML_OUTPUT_model", "outputs")
        os.makedirs(output_path, exist_ok=True)
        
        model_path = os.path.join(output_path, "quantile_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'model_p10': model.model_p10,
                'model_p50': model.model_p50,
                'model_p90': model.model_p90,
                'feature_columns': loader.feature_columns,
                'scaler': loader.scaler,
                'categorical_encoders': loader.categorical_encoders,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }, f)
        
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info("  Saved components:")
        logger.info("    - QuantileDARTModel (complete model)")
        logger.info("    - P10 LightGBM model (10th percentile)")
        logger.info("    - P50 LightGBM model (50th percentile/median)")
        logger.info("    - P90 LightGBM model (90th percentile)")
        logger.info("    - Feature preprocessing")
        logger.info("    - Performance metrics")
        
        logger.info("\n" + "="*80)
        logger.info("QUANTILE DART TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Test MAE (P50):  ${test_metrics['mae_p50']:.2f}")
        logger.info(f"Test RMSE (P50): ${test_metrics['rmse_p50']:.2f}")
        logger.info(f"Test R² (P50):   {test_metrics['r2_p50']:.4f}")
        logger.info(f"\nTest Coverage (P10-P90): {test_metrics['coverage_80']:.4f} ({test_metrics['coverage_80']*100:.2f}%) [Target: 80%]")
        logger.info(f"Mean Interval Width: ${test_metrics['interval_mean_width']:.2f}")
        logger.info(f"\nThis model provides:")
        logger.info("  - Point prediction (P50/median)")
        logger.info("  - 80% prediction interval (P10-P90)")
        logger.info("  - Uncertainty quantification for every prediction")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

