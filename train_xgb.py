"""
train_xgb.py

Train XGBoost model to predict ERCOT DART spread.
"""

import os
import sys
import pickle
import logging
from datetime import datetime

import numpy as np
import xgboost as xgb

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model, log_metrics_to_mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost regressor with optimal hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained XGBoost model
    """
    logger.info("="*80)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("="*80)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Hyperparameters optimized for regression
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Train with early stopping
    logger.info("\nTraining with early stopping...")
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    logger.info(f"✓ Training complete. Best iteration: {model.best_iteration}")
    
    return model


def main():
    """Main training pipeline."""
    try:
        logger.info("="*80)
        logger.info("XGBOOST TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Train model
        model = train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        dtrain = xgb.DMatrix(X_train)
        dval = xgb.DMatrix(X_val)
        dtest = xgb.DMatrix(X_test)
        
        y_train_pred = model.predict(dtrain)
        train_metrics = evaluate_model(y_train, y_train_pred, "Train")
        log_metrics_to_mlflow(train_metrics, "train")
        
        y_val_pred = model.predict(dval)
        val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
        log_metrics_to_mlflow(val_metrics, "val")
        
        y_test_pred = model.predict(dtest)
        test_metrics = evaluate_model(y_test, y_test_pred, "Test")
        log_metrics_to_mlflow(test_metrics, "test")
        
        # Feature importance
        logger.info("\n" + "="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES")
        logger.info("="*80)
        
        importance_dict = model.get_score(importance_type='gain')
        feature_names = loader.feature_columns
        
        # Map feature indices to names
        importance_list = []
        for i in range(len(feature_names)):
            feat_key = f'f{i}'
            if feat_key in importance_dict:
                importance_list.append((feature_names[i], importance_dict[feat_key]))
        
        importance_list = sorted(importance_list, key=lambda x: x[1], reverse=True)[:20]
        
        for i, (name, importance) in enumerate(importance_list, 1):
            logger.info(f"  {i:2d}. {name:50s} {importance:10.2f}")
        
        # Save model
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        # Get output path from Azure ML or use local
        output_path = os.environ.get("AZUREML_OUTPUT_model", "outputs")
        os.makedirs(output_path, exist_ok=True)
        
        model_path = os.path.join(output_path, "xgb_model.pkl")
        
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
        logger.info("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

