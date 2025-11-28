"""
metrics.py

Evaluation metrics for ERCOT DART spread prediction models.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)


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


# ============================================================================
# Classification Metrics
# ============================================================================

def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "Test") -> Dict:
    """
    Calculate all evaluation metrics for a classification model.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        dataset_name: Name of the dataset (for logging)
    
    Returns:
        Dictionary of metrics including accuracy, F1, and confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix
    }
    
    print(f"\n{dataset_name} Set Classification Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 (Macro):  {f1_macro:.4f}")
    print(f"  F1 (Weighted): {f1_weighted:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {conf_matrix}")
    
    # Detailed classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Regime 0', 'Regime 1', 'Regime 2', 'Regime 3']))
    
    return metrics


def create_regime_labels(dart_values: np.ndarray) -> np.ndarray:
    """
    Create regime labels based on absolute DART spread magnitude.
    
    Regime definitions:
        0: |DART| < 10
        1: 10 <= |DART| < 50
        2: 50 <= |DART| < 100
        3: |DART| >= 100
    
    Args:
        dart_values: Array of DART spread values
    
    Returns:
        Array of regime labels (0, 1, 2, or 3)
    """
    abs_dart = np.abs(dart_values)
    
    regime = np.zeros(len(dart_values), dtype=int)
    regime[abs_dart >= 10] = 1
    regime[abs_dart >= 50] = 2
    regime[abs_dart >= 100] = 3
    
    return regime


def print_regime_distribution(regime_labels: np.ndarray, dataset_name: str = "Dataset"):
    """
    Print the distribution of regime labels.
    
    Args:
        regime_labels: Array of regime labels
        dataset_name: Name of the dataset
    """
    unique, counts = np.unique(regime_labels, return_counts=True)
    total = len(regime_labels)
    
    print(f"\n{dataset_name} Regime Distribution:")
    for regime, count in zip(unique, counts):
        pct = (count / total) * 100
        regime_desc = {
            0: "|DART| < 10",
            1: "10 <= |DART| < 50",
            2: "50 <= |DART| < 100",
            3: "|DART| >= 100"
        }.get(regime, "Unknown")
        print(f"  Regime {regime} ({regime_desc}): {count:,} ({pct:.2f}%)")


# ============================================================================
# Quantile Regression Metrics
# ============================================================================

def calculate_quantile_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate empirical coverage of prediction interval.
    
    For a P10-P90 interval, we expect 80% coverage (80% of actual values fall within interval).
    
    Args:
        y_true: True values
        y_lower: Lower quantile predictions (e.g., P10)
        y_upper: Upper quantile predictions (e.g., P90)
    
    Returns:
        Empirical coverage as a fraction (0.0 to 1.0)
    """
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = np.mean(in_interval)
    return coverage


def calculate_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about prediction interval widths.
    
    Args:
        y_lower: Lower quantile predictions
        y_upper: Upper quantile predictions
    
    Returns:
        Dictionary with mean, median, std of interval widths
    """
    widths = y_upper - y_lower
    
    return {
        'mean': np.mean(widths),
        'median': np.median(widths),
        'std': np.std(widths),
        'min': np.min(widths),
        'max': np.max(widths)
    }


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Calculate quantile loss (pinball loss).
    
    Args:
        y_true: True values
        y_pred: Predicted quantile values
        alpha: Quantile level (e.g., 0.1 for 10th percentile)
    
    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)
    return np.mean(loss)


def evaluate_quantile_model(
    y_true: np.ndarray, 
    y_pred_dict: Dict[str, np.ndarray], 
    dataset_name: str = "Test"
) -> Dict:
    """
    Evaluate quantile regression model with P10, P50, P90 predictions.
    
    Args:
        y_true: True values
        y_pred_dict: Dictionary with keys 'p10', 'p50', 'p90'
        dataset_name: Name of the dataset (for logging)
    
    Returns:
        Dictionary of metrics
    """
    p10 = y_pred_dict['p10']
    p50 = y_pred_dict['p50']
    p90 = y_pred_dict['p90']
    
    # Point prediction metrics (using P50/median)
    mae_p50 = calculate_mae(y_true, p50)
    rmse_p50 = calculate_rmse(y_true, p50)
    r2_p50 = calculate_r2(y_true, p50)
    
    # Quantile-specific metrics
    coverage_80 = calculate_quantile_coverage(y_true, p10, p90)
    interval_stats = calculate_interval_width(p10, p90)
    
    # Quantile losses
    loss_p10 = quantile_loss(y_true, p10, 0.1)
    loss_p50 = quantile_loss(y_true, p50, 0.5)
    loss_p90 = quantile_loss(y_true, p90, 0.9)
    
    # Check quantile ordering (P10 <= P50 <= P90)
    violations_10_50 = np.sum(p10 > p50)
    violations_50_90 = np.sum(p50 > p90)
    violations_total = violations_10_50 + violations_50_90
    violations_pct = (violations_total / (2 * len(y_true))) * 100
    
    metrics = {
        'mae_p50': mae_p50,
        'rmse_p50': rmse_p50,
        'r2_p50': r2_p50,
        'coverage_80': coverage_80,
        'interval_mean_width': interval_stats['mean'],
        'interval_median_width': interval_stats['median'],
        'quantile_loss_p10': loss_p10,
        'quantile_loss_p50': loss_p50,
        'quantile_loss_p90': loss_p90,
        'quantile_violations': violations_total,
        'quantile_violations_pct': violations_pct
    }
    
    print(f"\n{dataset_name} Set Quantile Regression Metrics:")
    print(f"\n  Point Prediction (P50/Median):")
    print(f"    MAE:  ${mae_p50:.4f}")
    print(f"    RMSE: ${rmse_p50:.4f}")
    print(f"    R²:   {r2_p50:.4f}")
    
    print(f"\n  Prediction Interval (P10-P90):")
    print(f"    Coverage:     {coverage_80:.4f} ({coverage_80*100:.2f}%) [Target: 80%]")
    print(f"    Mean Width:   ${interval_stats['mean']:.2f}")
    print(f"    Median Width: ${interval_stats['median']:.2f}")
    
    print(f"\n  Quantile Calibration:")
    print(f"    P10 Loss: {loss_p10:.4f}")
    print(f"    P50 Loss: {loss_p50:.4f}")
    print(f"    P90 Loss: {loss_p90:.4f}")
    
    print(f"\n  Quantile Ordering:")
    print(f"    Violations: {violations_total:,} / {2*len(y_true):,} ({violations_pct:.2f}%)")
    if violations_pct > 0:
        print(f"    ⚠️  Some predictions violate P10 <= P50 <= P90 ordering")
    else:
        print(f"    ✓ All predictions maintain P10 <= P50 <= P90 ordering")
    
    # Coverage interpretation
    coverage_diff = abs(coverage_80 - 0.80) * 100
    if coverage_diff < 2:
        print(f"\n  ✓ Excellent calibration: Coverage within 2% of target")
    elif coverage_diff < 5:
        print(f"\n  ⚠️  Good calibration: Coverage within 5% of target")
    else:
        if coverage_80 < 0.80:
            print(f"\n  ⚠️  Under-coverage: Intervals too narrow")
        else:
            print(f"\n  ⚠️  Over-coverage: Intervals too wide")
    
    return metrics

