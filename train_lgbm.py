"""
train_lgbm.py

Train LightGBM model to predict ERCOT DART spread.
"""

import os
import sys
import pickle
import logging
from datetime import datetime

import numpy as np
import lightgbm as lgb

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model, log_metrics_to_mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM regressor with optimal hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained LightGBM model
    """
    logger.info("="*80)
    logger.info("TRAINING LIGHTGBM MODEL")
    logger.info("="*80)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Hyperparameters optimized for regression
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_jobs': -1
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Train with early stopping
    logger.info("\nTraining with early stopping...")
    
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    logger.info(f"✓ Training complete. Best iteration: {model.best_iteration}")
    
    return model


def main():
    """Main training pipeline."""
    try:
        logger.info("="*80)
        logger.info("LIGHTGBM TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Train model
        model = train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        train_metrics = evaluate_model(y_train, y_train_pred, "Train")
        log_metrics_to_mlflow(train_metrics, "train")
        
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
        log_metrics_to_mlflow(val_metrics, "val")
        
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_metrics = evaluate_model(y_test, y_test_pred, "Test")
        log_metrics_to_mlflow(test_metrics, "test")
        
        # Feature importance
        logger.info("\n" + "="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES")
        logger.info("="*80)
        
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = loader.feature_columns
        
        importance_df = sorted(
            zip(feature_names, feature_importance),
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
        
        model_path = os.path.join(output_path, "lgbm_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
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
        
        logger.info("\n" + "="*80)
        logger.info("LIGHTGBM TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

