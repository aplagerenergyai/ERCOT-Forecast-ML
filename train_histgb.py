"""
train_histgb.py

HistGradientBoosting Regressor for DART spread prediction.
Sklearn's native gradient boosting, similar to LightGBM but pure Python.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_histgb(X_train, y_train, X_val, y_val):
    """
    Train HistGradientBoosting Regressor with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
    
    Returns:
        Trained HistGradientBoosting model
    """
    # HistGradientBoosting parameters
    params = {
        'max_iter': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_samples_leaf': 20,
        'l2_regularization': 1.0,
        'early_stopping': True,
        'validation_fraction': None,  # We provide validation set
        'n_iter_no_change': 50,
        'random_state': 42,
        'verbose': 1,
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Train model with early stopping
    logger.info("Training with early stopping...")
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    logger.info(f"✓ Training complete. Best iteration: {model.n_iter_}")
    
    return model


def main():
    logger.info("="*80)
    logger.info("HISTGRADIENTBOOSTING TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Train model
        logger.info("="*80)
        logger.info("TRAINING HISTGRADIENTBOOSTING MODEL")
        logger.info("="*80)
        model = train_histgb(X_train, y_train, X_val, y_val)
        
        # Evaluate
        logger.info("")
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        logger.info("")
        logger.info("Train Set Metrics:")
        evaluate_model(y_train, y_train_pred)
        
        logger.info("")
        logger.info("Validation Set Metrics:")
        evaluate_model(y_val, y_val_pred)
        
        logger.info("")
        logger.info("Test Set Metrics:")
        evaluate_model(y_test, y_test_pred)
        
        # Save model
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        output_path = "./outputs/histgb_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("HISTGRADIENTBOOSTING TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

