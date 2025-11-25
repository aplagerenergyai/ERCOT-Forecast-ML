"""
train_catboost.py

CatBoost Regressor for DART spread prediction.
Better handling of categorical features than LightGBM/XGBoost.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
from catboost import CatBoostRegressor, Pool

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_catboost(X_train, y_train, X_val, y_val, categorical_features=None):
    """
    Train CatBoost Regressor with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        categorical_features: List of categorical feature indices
    
    Returns:
        Trained CatBoost model
    """
    # CatBoost parameters
    params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'task_type': 'GPU',  # GPU acceleration enabled
        'devices': '0',  # Use first GPU
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Create CatBoost pools
    # Note: categorical_features is None because data is pre-encoded
    if categorical_features:
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    else:
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
    
    # Train model
    logger.info("Training with early stopping...")
    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )
    
    logger.info(f"✓ Training complete. Best iteration: {model.best_iteration_}")
    
    return model


def main():
    logger.info("="*80)
    logger.info("CATBOOST TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Note: Categorical features are already encoded by dataloader (TargetEncoder)
        # CatBoost will treat all features as numeric
        logger.info("Note: Categorical features pre-encoded by TargetEncoder")
        logger.info(f"  Original categorical columns: {loader.categorical_columns}")
        
        # Train model
        logger.info("="*80)
        logger.info("TRAINING CATBOOST MODEL")
        logger.info("="*80)
        model = train_catboost(X_train, y_train, X_val, y_val, categorical_features=None)
        
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
        feature_importance = model.get_feature_importance()
        feature_names = loader.feature_columns
        importance_df = list(zip(feature_names, feature_importance))
        importance_df = sorted(importance_df, key=lambda x: x[1], reverse=True)[:20]
        
        for idx, (feature, importance) in enumerate(importance_df, 1):
            logger.info(f"  {idx:2d}. {feature:50s} {importance:,.2f}")
        
        # Save model
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        output_path = "./outputs/catboost_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("CATBOOST TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

