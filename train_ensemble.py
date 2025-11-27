"""
train_ensemble.py

Ensemble model combining LightGBM, XGBoost, CatBoost, RandomForest, and Deep Learning.
Uses simple averaging (voting) or weighted averaging based on validation performance.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load a pickled model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"  ✓ Loaded: {os.path.basename(model_path)}")
        return model
    except FileNotFoundError:
        logger.warning(f"  ✗ Not found: {os.path.basename(model_path)}")
        return None


def simple_ensemble(predictions_list, weights=None):
    """
    Create ensemble predictions using weighted average.
    
    Args:
        predictions_list: List of prediction arrays from different models
        weights: Optional weights for each model (default: equal weights)
    
    Returns:
        Ensemble predictions
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Weighted average
    ensemble_pred = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += pred * weight
    
    return ensemble_pred


def main():
    logger.info("="*80)
    logger.info("ENSEMBLE MODEL TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        logger.info("="*80)
        logger.info("LOADING INDIVIDUAL MODELS")
        logger.info("="*80)
        
        # Try to load all available models
        model_paths = {
            'lgbm': './outputs/lgbm_model.pkl',
            'xgb': './outputs/xgb_model.pkl',
            'catboost': './outputs/catboost_model.pkl',
            'random_forest': './outputs/random_forest_model.pkl',
            'deep': './outputs/deep_model.pt',  # PyTorch model
        }
        
        models = {}
        for name, path in model_paths.items():
            model = load_model(path)
            if model is not None:
                models[name] = model
        
        if len(models) < 2:
            raise ValueError(f"Need at least 2 models for ensemble, found {len(models)}")
        
        logger.info(f"✓ Loaded {len(models)} models for ensemble")
        
        # Generate predictions from each model
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING PREDICTIONS FROM INDIVIDUAL MODELS")
        logger.info("="*80)
        
        val_predictions = {}
        test_predictions = {}
        val_scores = {}
        
        for name, model in models.items():
            logger.info(f"  Predicting with {name}...")
            
            # For PyTorch models, handle differently
            if name == 'deep' and hasattr(model, 'eval'):
                import torch
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val.values)
                    X_test_tensor = torch.FloatTensor(X_test.values)
                    val_pred = model(X_val_tensor).numpy().flatten()
                    test_pred = model(X_test_tensor).numpy().flatten()
            else:
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            
            val_predictions[name] = val_pred
            test_predictions[name] = test_pred
            
            # Calculate validation RMSE for weighting
            rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_scores[name] = rmse
            logger.info(f"    Validation RMSE: ${rmse:.2f}")
        
        # Calculate weights based on inverse RMSE (better models get higher weight)
        logger.info("")
        logger.info("="*80)
        logger.info("CALCULATING ENSEMBLE WEIGHTS")
        logger.info("="*80)
        
        # Inverse RMSE weighting (lower RMSE = higher weight)
        inverse_rmse = {name: 1.0 / score for name, score in val_scores.items()}
        total_inverse = sum(inverse_rmse.values())
        weights = {name: inv / total_inverse for name, inv in inverse_rmse.items()}
        
        logger.info("Weights based on validation performance:")
        for name, weight in weights.items():
            logger.info(f"  {name:15s}: {weight:.4f} (Val RMSE: ${val_scores[name]:.2f})")
        
        # Create ensemble predictions
        logger.info("")
        logger.info("="*80)
        logger.info("CREATING ENSEMBLE PREDICTIONS")
        logger.info("="*80)
        
        val_pred_list = [val_predictions[name] for name in models.keys()]
        test_pred_list = [test_predictions[name] for name in models.keys()]
        weight_list = [weights[name] for name in models.keys()]
        
        y_val_ensemble = simple_ensemble(val_pred_list, weight_list)
        y_test_ensemble = simple_ensemble(test_pred_list, weight_list)
        
        logger.info("✓ Ensemble predictions created using weighted average")
        
        # Evaluate ensemble
        logger.info("")
        logger.info("="*80)
        logger.info("ENSEMBLE MODEL EVALUATION")
        logger.info("="*80)
        
        logger.info("")
        logger.info("Validation Set Metrics:")
        evaluate_model(y_val, y_val_ensemble)
        
        logger.info("")
        logger.info("Test Set Metrics:")
        evaluate_model(y_test, y_test_ensemble)
        
        # Save ensemble configuration
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING ENSEMBLE CONFIGURATION")
        logger.info("="*80)
        output_path = "./outputs/ensemble_config.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        ensemble_config = {
            'models': list(models.keys()),
            'weights': weights,
            'val_scores': val_scores,
            'ensemble_val_rmse': np.sqrt(np.mean((y_val - y_val_ensemble) ** 2)),
            'ensemble_test_rmse': np.sqrt(np.mean((y_test - y_test_ensemble) ** 2)),
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(ensemble_config, f)
        logger.info(f"✓ Ensemble config saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("ENSEMBLE TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise


if __name__ == "__main__":
    main()

