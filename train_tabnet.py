"""
train_tabnet.py

TabNet for DART spread prediction.
Attention-based deep learning for tabular data with built-in interpretability.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_tabnet(X_train, y_train, X_val, y_val):
    """
    Train TabNet Regressor with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
    
    Returns:
        Trained TabNet model
    """
    # TabNet parameters
    params = {
        'n_d': 64,  # Width of decision prediction layer
        'n_a': 64,  # Width of attention embedding
        'n_steps': 5,  # Number of steps in the architecture
        'gamma': 1.5,  # Coefficient for feature reusage
        'n_independent': 2,  # Number of independent GLU layers
        'n_shared': 2,  # Number of shared GLU layers
        'lambda_sparse': 1e-4,  # Sparsity regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': {'lr': 2e-2},
        'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {'mode': 'min', 'patience': 10, 'factor': 0.5},
        'mask_type': 'entmax',
        'verbose': 10,
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    logger.info("Hyperparameters:")
    for key, value in params.items():
        if key not in ['optimizer_fn', 'scheduler_fn']:
            logger.info(f"  {key}: {value}")
    logger.info(f"  Using device: {params['device_name']}")
    logger.info("")
    
    # Initialize model
    model = TabNetRegressor(**params)
    
    # Train with early stopping
    logger.info("Training TabNet with early stopping...")
    model.fit(
        X_train, y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False,
    )
    
    logger.info(f"✓ Training complete. Best epoch: {model.best_epoch}")
    
    return model


def main():
    logger.info("="*80)
    logger.info("TABNET TRAINING PIPELINE")
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
        logger.info("TRAINING TABNET MODEL")
        logger.info("="*80)
        model = train_tabnet(X_train, y_train, X_val, y_val)
        
        # Evaluate
        logger.info("")
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train).flatten()
        y_val_pred = model.predict(X_val).flatten()
        y_test_pred = model.predict(X_test).flatten()
        
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
        logger.info("TOP 20 FEATURE IMPORTANCES (Attention-based)")
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
        output_path = "./outputs/tabnet_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        # Save using TabNet's save method
        model.save_model(output_path.replace('.pkl', ''))
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("TABNET TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

