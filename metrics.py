"""
metrics.py

Evaluation metrics for ERCOT DART spread prediction models.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero
    
    Returns:
        MAPE (as percentage)
    """
    # Avoid division by zero
    denominator = np.abs(y_true) + epsilon
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    return mape


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    return r2_score(y_true, y_pred)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "Test") -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dataset_name: Name of the dataset (for logging)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  RMSE: ${metrics['rmse']:.4f}")
    print(f"  MAE:  ${metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    return metrics


def log_metrics_to_mlflow(metrics: Dict[str, float], prefix: str = "test"):
    """
    Log metrics to MLflow (if available).
    
    Args:
        metrics: Dictionary of metric name -> value
        prefix: Prefix for metric names (e.g., "train", "val", "test")
    """
    try:
        import mlflow
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{prefix}_{metric_name}", value)
    
    except ImportError:
        pass  # MLflow not available

