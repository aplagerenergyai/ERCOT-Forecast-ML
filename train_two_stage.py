"""
train_two_stage.py

Train Two-Stage DART model:
  Stage 1: Predict DAM_Price_Hourly
  Stage 2: Predict RTM_LMP_HourlyAvg
  Final: DART_pred = RT_pred - DA_pred
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model, log_metrics_to_mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwoStageDARTModel(BaseEstimator, RegressorMixin):
    """
    Two-stage DART prediction model.
    
    Stage 1: Predict DA_LMP using LightGBM
    Stage 2: Predict RT_LMP using LightGBM
    Final Prediction: DART = RT_pred - DA_pred
    
    This class implements the standard scikit-learn interface:
    - .fit(X, y) for training
    - .predict(X) for prediction
    """
    
    def __init__(
        self,
        n_estimators=800,
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        min_data_in_leaf=20,
        verbose=-1,
        device='gpu'
    ):
        """Initialize the two-stage model with LightGBM hyperparameters."""
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.verbose = verbose
        self.device = device
        
        # Models will be trained in .fit()
        self.da_model = None
        self.rt_model = None
        
        # Store target values during fit
        self.y_da_train = None
        self.y_rt_train = None
    
    def _get_lgb_params(self):
        """Get LightGBM parameters."""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 5,
            'max_depth': self.max_depth,
            'min_data_in_leaf': self.min_data_in_leaf,
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
    
    def fit(self, X, y, X_val=None, y_val=None, y_da_train=None, y_rt_train=None, 
            y_da_val=None, y_rt_val=None):
        """
        Train both DA and RT models.
        
        Args:
            X: Feature matrix (training)
            y: Target DART spread (not used, but required for sklearn interface)
            X_val: Validation features (optional)
            y_val: Validation DART spread (not used, but for validation)
            y_da_train: Training DA prices
            y_rt_train: Training RT prices
            y_da_val: Validation DA prices (optional)
            y_rt_val: Validation RT prices (optional)
        """
        logger.info("="*80)
        logger.info("TWO-STAGE DART MODEL TRAINING")
        logger.info("="*80)
        
        if y_da_train is None or y_rt_train is None:
            raise ValueError("Must provide y_da_train and y_rt_train for two-stage training")
        
        # Store targets
        self.y_da_train = y_da_train
        self.y_rt_train = y_rt_train
        
        params = self._get_lgb_params()
        
        # ========================================================================
        # STAGE 1: Train DA Model
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: TRAINING DA PRICE MODEL")
        logger.info("="*80)
        
        train_data_da = lgb.Dataset(X, label=y_da_train)
        
        callbacks_da = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets_da = [train_data_da]
        valid_names_da = ['train']
        
        if X_val is not None and y_da_val is not None:
            val_data_da = lgb.Dataset(X_val, label=y_da_val, reference=train_data_da)
            valid_sets_da.append(val_data_da)
            valid_names_da.append('val')
        
        logger.info("Training DA model...")
        self.da_model = lgb.train(
            params,
            train_data_da,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_da,
            valid_names=valid_names_da,
            callbacks=callbacks_da
        )
        
        logger.info(f"✓ DA model trained. Best iteration: {self.da_model.best_iteration}")
        
        # ========================================================================
        # STAGE 2: Train RT Model
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: TRAINING RT PRICE MODEL")
        logger.info("="*80)
        
        train_data_rt = lgb.Dataset(X, label=y_rt_train)
        
        callbacks_rt = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets_rt = [train_data_rt]
        valid_names_rt = ['train']
        
        if X_val is not None and y_rt_val is not None:
            val_data_rt = lgb.Dataset(X_val, label=y_rt_val, reference=train_data_rt)
            valid_sets_rt.append(val_data_rt)
            valid_names_rt.append('val')
        
        logger.info("Training RT model...")
        self.rt_model = lgb.train(
            params,
            train_data_rt,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_rt,
            valid_names=valid_names_rt,
            callbacks=callbacks_rt
        )
        
        logger.info(f"✓ RT model trained. Best iteration: {self.rt_model.best_iteration}")
        
        logger.info("\n" + "="*80)
        logger.info("TWO-STAGE MODEL TRAINING COMPLETE")
        logger.info("="*80)
        
        return self
    
    def predict(self, X):
        """
        Predict DART spread as: RT_pred - DA_pred
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted DART spread
        """
        if self.da_model is None or self.rt_model is None:
            raise RuntimeError("Model must be trained before prediction. Call .fit() first.")
        
        # Predict DA and RT separately
        da_pred = self.da_model.predict(X, num_iteration=self.da_model.best_iteration)
        rt_pred = self.rt_model.predict(X, num_iteration=self.rt_model.best_iteration)
        
        # Compute DART spread
        dart_pred = rt_pred - da_pred
        
        return dart_pred
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance from both models."""
        if self.da_model is None or self.rt_model is None:
            raise RuntimeError("Model must be trained before getting feature importance.")
        
        da_importance = self.da_model.feature_importance(importance_type=importance_type)
        rt_importance = self.rt_model.feature_importance(importance_type=importance_type)
        
        # Average importance across both models
        avg_importance = (da_importance + rt_importance) / 2
        
        return {
            'da': da_importance,
            'rt': rt_importance,
            'average': avg_importance
        }


class TwoStageDataLoader(ERCOTDataLoader):
    """
    Extended data loader that also extracts DA and RT targets for two-stage training.
    """
    
    def prepare_datasets_two_stage(
        self, 
        train_pct: float = 0.8, 
        val_pct: float = 0.1, 
        test_pct: float = 0.1,
        sample_rows: int = None
    ):
        """
        Prepare datasets for two-stage training.
        
        Returns:
            (X_train, y_train, y_da_train, y_rt_train), 
            (X_val, y_val, y_da_val, y_rt_val), 
            (X_test, y_test, y_da_test, y_rt_test)
        """
        logger.info("="*80)
        logger.info("STARTING DATA PREPARATION PIPELINE (TWO-STAGE)")
        logger.info("="*80)
        
        # Load data
        df = self.load_data(sample_rows=sample_rows)
        
        # Create DART target (also validates DA/RT columns exist)
        df = self.create_target(df)
        
        # Extract DA and RT values BEFORE they're dropped
        da_values = df['DAM_Price_Hourly'].values
        rt_values = df['RTM_LMP_HourlyAvg'].values
        
        # Time-based split
        train_df, val_df, test_df = self.time_based_split(df, train_pct, val_pct, test_pct)
        
        # Get indices for splits
        train_indices = train_df.index
        val_indices = val_df.index
        test_indices = test_df.index
        
        # Extract DA and RT targets for each split
        y_da_train = da_values[train_indices]
        y_rt_train = rt_values[train_indices]
        
        y_da_val = da_values[val_indices]
        y_rt_val = rt_values[val_indices]
        
        y_da_test = da_values[test_indices]
        y_rt_test = rt_values[test_indices]
        
        # Identify features
        feature_info = self.identify_feature_columns(train_df)
        self.feature_columns = feature_info['all']
        self.categorical_columns = feature_info['categorical']
        self.continuous_columns = feature_info['continuous']
        
        # Encode categorical features
        train_df, val_df, test_df = self.encode_categorical_features(
            train_df, val_df, test_df, self.categorical_columns
        )
        
        # Standardize continuous features
        train_df, val_df, test_df = self.standardize_continuous_features(
            train_df, val_df, test_df, self.continuous_columns
        )
        
        # Extract X and y (DART)
        X_train = train_df[self.feature_columns].values
        y_train = train_df['DART'].values
        
        X_val = val_df[self.feature_columns].values
        y_val = val_df['DART'].values
        
        X_test = test_df[self.feature_columns].values
        y_test = test_df['DART'].values
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION COMPLETE (TWO-STAGE)")
        logger.info("="*80)
        logger.info(f"Train: X={X_train.shape}, y_DART={y_train.shape}, y_DA={y_da_train.shape}, y_RT={y_rt_train.shape}")
        logger.info(f"Val:   X={X_val.shape}, y_DART={y_val.shape}, y_DA={y_da_val.shape}, y_RT={y_rt_val.shape}")
        logger.info(f"Test:  X={X_test.shape}, y_DART={y_test.shape}, y_DA={y_da_test.shape}, y_RT={y_rt_test.shape}")
        
        return (
            (X_train, y_train, y_da_train, y_rt_train),
            (X_val, y_val, y_da_val, y_rt_val),
            (X_test, y_test, y_da_test, y_rt_test)
        )


def main():
    """Main training pipeline."""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Train Two-Stage DART model')
        parser.add_argument('--features_path', type=str, default=None, 
                            help='Path to features directory or parquet file')
        args = parser.parse_args()
        
        logger.info("="*80)
        logger.info("TWO-STAGE DART TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        if args.features_path:
            features_path = os.path.join(args.features_path, "hourly_features.parquet")
            logger.info(f"Using command-line features path: {features_path}")
        else:
            features_path = load_features_from_aml_input("features")
        
        # Load and prepare data with two-stage loader
        loader = TwoStageDataLoader(features_path)
        (X_train, y_train, y_da_train, y_rt_train), \
        (X_val, y_val, y_da_val, y_rt_val), \
        (X_test, y_test, y_da_test, y_rt_test) = loader.prepare_datasets_two_stage()
        
        # Initialize and train model
        model = TwoStageDARTModel(
            n_estimators=800,
            num_leaves=64,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            device='gpu'
        )
        
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            y_da_train=y_da_train, y_rt_train=y_rt_train,
            y_da_val=y_da_val, y_rt_val=y_rt_val
        )
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train)
        train_metrics = evaluate_model(y_train, y_train_pred, "Train")
        log_metrics_to_mlflow(train_metrics, "train")
        
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
        log_metrics_to_mlflow(val_metrics, "val")
        
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_test_pred, "Test")
        log_metrics_to_mlflow(test_metrics, "test")
        
        # Component-wise evaluation
        logger.info("\n" + "="*80)
        logger.info("COMPONENT-WISE EVALUATION (DA and RT Predictions)")
        logger.info("="*80)
        
        # DA predictions
        y_da_train_pred = model.da_model.predict(X_train, num_iteration=model.da_model.best_iteration)
        y_da_val_pred = model.da_model.predict(X_val, num_iteration=model.da_model.best_iteration)
        y_da_test_pred = model.da_model.predict(X_test, num_iteration=model.da_model.best_iteration)
        
        logger.info("\nDA Price Predictions:")
        evaluate_model(y_da_train, y_da_train_pred, "Train (DA)")
        evaluate_model(y_da_val, y_da_val_pred, "Val (DA)")
        evaluate_model(y_da_test, y_da_test_pred, "Test (DA)")
        
        # RT predictions
        y_rt_train_pred = model.rt_model.predict(X_train, num_iteration=model.rt_model.best_iteration)
        y_rt_val_pred = model.rt_model.predict(X_val, num_iteration=model.rt_model.best_iteration)
        y_rt_test_pred = model.rt_model.predict(X_test, num_iteration=model.rt_model.best_iteration)
        
        logger.info("\nRT Price Predictions:")
        evaluate_model(y_rt_train, y_rt_train_pred, "Train (RT)")
        evaluate_model(y_rt_val, y_rt_val_pred, "Val (RT)")
        evaluate_model(y_rt_test, y_rt_test_pred, "Test (RT)")
        
        # Feature importance
        logger.info("\n" + "="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES (AVERAGED ACROSS DA AND RT)")
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
        
        model_path = os.path.join(output_path, "two_stage_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'da_model': model.da_model,
                'rt_model': model.rt_model,
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
        logger.info("    - TwoStageDARTModel (complete model)")
        logger.info("    - DA LightGBM model")
        logger.info("    - RT LightGBM model")
        logger.info("    - Feature preprocessing")
        logger.info("    - Performance metrics")
        
        logger.info("\n" + "="*80)
        logger.info("TWO-STAGE DART TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

