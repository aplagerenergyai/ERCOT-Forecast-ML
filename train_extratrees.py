"""
train_extratrees.py

Extra Trees Regressor for DART spread prediction.
More randomized than Random Forest, often better on large datasets.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_extratrees(X_train, y_train, X_val, y_val):
    """
    Train Extra Trees Regressor.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for monitoring only)
    
    Returns:
        Trained Extra Trees model
    """
    # Extra Trees parameters (more randomized than Random Forest)
    params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'max_features': 'sqrt',
        'bootstrap': False,  # Key difference from RF
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 1,
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Train model
    logger.info("Training Extra Trees...")
    logger.info("Note: Extra Trees uses more randomization than Random Forest")
    logger.info("Training all 200 trees...")
    
    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"✓ Training complete. Trained {model.n_estimators} trees")
    
    return model


def main():
    logger.info("="*80)
    logger.info("EXTRA TREES TRAINING PIPELINE")
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
        logger.info("TRAINING EXTRA TREES MODEL")
        logger.info("="*80)
        model = train_extratrees(X_train, y_train, X_val, y_val)
        
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
        
        # Feature importances
        logger.info("")
        logger.info("="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES")
        logger.info("="*80)
        feature_importance = model.feature_importances_
        feature_names = loader.feature_columns
        importance_df = list(zip(feature_names, feature_importance))
        importance_df = sorted(importance_df, key=lambda x: x[1], reverse=True)[:20]
        
        for idx, (feature, importance) in enumerate(importance_df, 1):
            logger.info(f"  {idx:2d}. {feature:50s} {importance:.6f}")
        
        # Save model
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        output_path = "./outputs/extratrees_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("EXTRA TREES TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

