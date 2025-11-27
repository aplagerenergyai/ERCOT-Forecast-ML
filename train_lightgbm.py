"""
train_lightgbm.py

Trains a LightGBM regressor on ERCOT price forecasting data.
Compatible with Azure ML pipelines.
"""

import os
import argparse
import logging
from typing import Tuple

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the features dataset from Parquet file.
    
    Args:
        data_path: Path to the features.parquet file
    
    Returns:
        DataFrame with features
    """
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target price column.
    
    Args:
        df: Feature DataFrame
    
    Returns:
        Name of target column
    """
    # First, check if 'target_price' column exists
    if 'target_price' in df.columns:
        return 'target_price'
    
    # Otherwise, look for LMP price columns
    potential_targets = [
        col for col in df.columns 
        if 'lmp' in col.lower() and any(keyword in col.lower() 
            for keyword in ['price', 'settlementpointprice', 'spp'])
    ]
    
    if not potential_targets:
        # Fallback: use first column with 'lmp' in name
        potential_targets = [col for col in df.columns if 'lmp' in col.lower()]
    
    if not potential_targets:
        raise ValueError(
            "Could not identify target price column. "
            "No 'target_price' or 'lmp' columns found."
        )
    
    target_col = potential_targets[0]
    logger.info(f"Identified target column: {target_col}")
    
    return target_col


def prepare_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.
    
    Args:
        df: Feature DataFrame
        target_col: Name of target column
    
    Returns:
        Tuple of (X, y) where X contains all numeric features except target
    """
    logger.info("Preparing features and target...")
    
    # Extract target
    y = df[target_col].copy()
    
    # Get all numeric columns except target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    X = df[feature_cols].copy()
    
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target: {target_col}")
    logger.info(f"Sample features: {feature_cols[:5]}")
    
    # Check for missing values
    missing_in_X = X.isna().sum().sum()
    missing_in_y = y.isna().sum()
    
    if missing_in_X > 0:
        logger.warning(f"Found {missing_in_X} missing values in features - will be handled by LightGBM")
    
    if missing_in_y > 0:
        raise ValueError(f"Found {missing_in_y} missing values in target - this should not happen")
    
    return X, y


def chronological_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_ratio: Proportion of data for training
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Performing {int(train_ratio*100)}/{int((1-train_ratio)*100)} chronological split...")
    
    # Ensure data is sorted by timestamp (index)
    if not X.index.equals(y.index):
        raise ValueError("X and y indices do not match")
    
    # Sort by index (timestamp)
    sort_idx = X.index.argsort()
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    
    # Calculate split point
    split_idx = int(len(X) * train_ratio)
    
    # Split data
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train):,} samples ({X_train.index.min()} to {X_train.index.max()})")
    logger.info(f"Test set:  {len(X_test):,} samples ({X_test.index.min()} to {X_test.index.max()})")
    
    return X_train, X_test, y_train, y_test


def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    """
    Train a LightGBM regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained LGBMRegressor model
    """
    logger.info("Training LightGBM model...")
    
    # Initialize model with default parameters
    # LightGBM handles missing values internally
    model = LGBMRegressor(
        random_state=42,
        verbose=-1  # Suppress training output for Azure ML compatibility
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("✓ Model training completed")
    
    return model


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics: MAE, RMSE, MAPE.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def evaluate_model(model: LGBMRegressor, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate model on train and test sets and print metrics.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    """
    logger.info("Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Print metrics to stdout (required for Azure ML logging)
    print("\n" + "=" * 60)
    print("LIGHTGBM MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nTRAIN METRICS:")
    print(f"  MAE:  {train_metrics['MAE']:.4f}")
    print(f"  RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  MAPE: {train_metrics['MAPE']:.4f}%")
    
    print("\nTEST METRICS:")
    print(f"  MAE:  {test_metrics['MAE']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.4f}%")
    
    print("\n" + "=" * 60)
    
    # Also log to logger
    logger.info(f"Train - MAE: {train_metrics['MAE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}, MAPE: {train_metrics['MAPE']:.4f}%")
    logger.info(f"Test  - MAE: {test_metrics['MAE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, MAPE: {test_metrics['MAPE']:.4f}%")


def save_model(model: LGBMRegressor, output_dir: str = "model_output") -> str:
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model
        output_dir: Directory to save model
    
    Returns:
        Path to saved model file
    """
    logger.info(f"Saving model to {output_dir}/...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "lightgbm_model.pkl")
    joblib.dump(model, model_path)
    
    logger.info(f"✓ Model saved to: {model_path}")
    
    return model_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train LightGBM model for ERCOT price forecasting')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to features.parquet file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='model_output',
        help='Directory to save trained model (default: model_output)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (default: 0.8)'
    )
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("STARTING LIGHTGBM TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        df = load_data(args.data_path)
        
        # Identify target column
        target_col = identify_target_column(df)
        
        # Prepare features and target
        X, y = prepare_features_and_target(df, target_col)
        
        # Chronological split
        X_train, X_test, y_train, y_test = chronological_split(X, y, args.train_ratio)
        
        # Train model
        model = train_lightgbm_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Save model
        model_path = save_model(model, args.output_dir)
        
        logger.info("=" * 80)
        logger.info("LIGHTGBM TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

