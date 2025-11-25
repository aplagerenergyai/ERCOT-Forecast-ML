"""
train_ngboost.py

NGBoost for DART spread prediction.
Probabilistic gradient boosting that provides prediction intervals (uncertainty).
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_ngboost(X_train, y_train, X_val, y_val):
    """
    Train NGBoost Regressor with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
    
    Returns:
        Trained NGBoost model
    """
    # NGBoost parameters
    params = {
        'Dist': Normal,  # Normal distribution for regression
        'Base': DecisionTreeRegressor(
            criterion='friedman_mse',
            max_depth=5,
            min_samples_split=100,
            min_samples_leaf=50,
        ),
        'n_estimators': 500,
        'learning_rate': 0.05,
        'minibatch_frac': 0.5,
        'verbose': True,
        'verbose_eval': 50,
        'random_state': 42,
    }
    
    logger.info("Hyperparameters:")
    logger.info(f"  Distribution: Normal")
    logger.info(f"  n_estimators: {params['n_estimators']}")
    logger.info(f"  learning_rate: {params['learning_rate']}")
    logger.info(f"  minibatch_frac: {params['minibatch_frac']}")
    logger.info(f"  Base learner: DecisionTreeRegressor(max_depth=5)")
    logger.info("")
    
    # Initialize model
    model = NGBRegressor(**params)
    
    # Train with early stopping
    logger.info("Training NGBoost with early stopping...")
    model.fit(
        X_train.values, y_train.values,
        X_val=X_val.values, Y_val=y_val.values,
        early_stopping_rounds=50
    )
    
    logger.info(f"✓ Training complete. Best iteration: {model.best_val_loss_itr}")
    
    return model


def main():
    logger.info("="*80)
    logger.info("NGBOOST TRAINING PIPELINE")
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
        logger.info("TRAINING NGBOOST MODEL")
        logger.info("="*80)
        model = train_ngboost(X_train, y_train, X_val, y_val)
        
        # Evaluate
        logger.info("")
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        # Get predictions (point estimates)
        y_train_pred = model.predict(X_train.values)
        y_val_pred = model.predict(X_val.values)
        y_test_pred = model.predict(X_test.values)
        
        logger.info("")
        logger.info("Train Set Metrics:")
        evaluate_model(y_train, y_train_pred)
        
        logger.info("")
        logger.info("Validation Set Metrics:")
        evaluate_model(y_val, y_val_pred)
        
        logger.info("")
        logger.info("Test Set Metrics:")
        evaluate_model(y_test, y_test_pred)
        
        # Get prediction intervals (95% confidence)
        logger.info("")
        logger.info("="*80)
        logger.info("PREDICTION INTERVALS (95% Confidence)")
        logger.info("="*80)
        
        Y_dists = model.pred_dist(X_test.values)
        intervals = Y_dists.interval(alpha=0.05)  # 95% CI
        lower = intervals[0]
        upper = intervals[1]
        
        # Calculate average interval width
        avg_width = np.mean(upper - lower)
        logger.info(f"Average prediction interval width: ${avg_width:.2f}")
        
        # Calculate coverage (what % of actual values fall within intervals)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        logger.info(f"Interval coverage: {coverage*100:.2f}%")
        
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
        output_path = "./outputs/ngboost_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("NGBOOST TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

