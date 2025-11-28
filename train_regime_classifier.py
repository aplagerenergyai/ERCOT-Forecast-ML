"""
train_regime_classifier.py

Train LightGBM Classifier to predict DART spread regime (magnitude bucket).

Regime definitions:
    0: |DART| < 10
    1: 10 <= |DART| < 50
    2: 50 <= |DART| < 100
    3: |DART| >= 100
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import (
    evaluate_classifier, 
    log_metrics_to_mlflow,
    create_regime_labels,
    print_regime_distribution
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeClassifierModel(BaseEstimator, ClassifierMixin):
    """
    LightGBM classifier for predicting DART spread regime.
    
    Predicts which magnitude bucket a DART spread falls into:
        0: |DART| < 10         (Low spread)
        1: 10 <= |DART| < 50   (Medium spread)
        2: 50 <= |DART| < 100  (High spread)
        3: |DART| >= 100       (Very high spread)
    
    Implements standard scikit-learn classifier interface:
    - .fit(X, y) for training
    - .predict(X) for prediction
    - .predict_proba(X) for probability estimates
    """
    
    def __init__(
        self,
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        min_data_in_leaf=20,
        verbose=-1,
        device='gpu',
        num_class=4
    ):
        """Initialize the regime classifier with LightGBM hyperparameters."""
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.verbose = verbose
        self.device = device
        self.num_class = num_class
        
        # Model will be trained in .fit()
        self.model = None
        self.classes_ = np.array([0, 1, 2, 3])
    
    def _get_lgb_params(self):
        """Get LightGBM parameters for multiclass classification."""
        params = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 5,
            'max_depth': self.max_depth,
            'min_data_in_leaf': self.min_data_in_leaf,
            'verbose': self.verbose,
        }
        
        # Add GPU params if device is GPU
        if self.device == 'gpu':
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        return params
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the regime classifier.
        
        Args:
            X: Feature matrix (training)
            y: Target regime labels (0, 1, 2, or 3)
            X_val: Validation features (optional)
            y_val: Validation regime labels (optional)
        """
        logger.info("="*80)
        logger.info("REGIME CLASSIFIER TRAINING")
        logger.info("="*80)
        
        params = self._get_lgb_params()
        
        logger.info("Hyperparameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y)
        
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        logger.info("\nTraining classifier with early stopping...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        logger.info(f"✓ Training complete. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X):
        """
        Predict regime labels.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted regime labels (0, 1, 2, or 3)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction. Call .fit() first.")
        
        # Get probabilities and convert to class predictions
        proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        predictions = np.argmax(proba, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, n_classes) with probability estimates
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction. Call .fit() first.")
        
        proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        return proba
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance from the model."""
        if self.model is None:
            raise RuntimeError("Model must be trained before getting feature importance.")
        
        return self.model.feature_importance(importance_type=importance_type)


class RegimeDataLoader(ERCOTDataLoader):
    """
    Extended data loader that creates regime labels from DART values.
    """
    
    def prepare_datasets_regime_classification(
        self, 
        train_pct: float = 0.8, 
        val_pct: float = 0.1, 
        test_pct: float = 0.1,
        sample_rows: int = None
    ):
        """
        Prepare datasets for regime classification.
        
        Returns:
            (X_train, y_train_regime), 
            (X_val, y_val_regime), 
            (X_test, y_test_regime),
            (y_train_dart, y_val_dart, y_test_dart)  # Original DART values for reference
        """
        logger.info("="*80)
        logger.info("STARTING DATA PREPARATION PIPELINE (REGIME CLASSIFICATION)")
        logger.info("="*80)
        
        # Load data
        df = self.load_data(sample_rows=sample_rows)
        
        # Create DART target
        df = self.create_target(df)
        
        # Extract DART values before any transformations
        dart_values = df['DART'].values
        
        # Time-based split
        train_df, val_df, test_df = self.time_based_split(df, train_pct, val_pct, test_pct)
        
        # Get indices for splits
        train_indices = train_df.index
        val_indices = val_df.index
        test_indices = test_df.index
        
        # Extract DART values for each split
        y_train_dart = dart_values[train_indices]
        y_val_dart = dart_values[val_indices]
        y_test_dart = dart_values[test_indices]
        
        # Create regime labels
        y_train_regime = create_regime_labels(y_train_dart)
        y_val_regime = create_regime_labels(y_val_dart)
        y_test_regime = create_regime_labels(y_test_dart)
        
        # Print regime distributions
        print_regime_distribution(y_train_regime, "Training")
        print_regime_distribution(y_val_regime, "Validation")
        print_regime_distribution(y_test_regime, "Test")
        
        # Identify features
        feature_info = self.identify_feature_columns(train_df)
        self.feature_columns = feature_info['all']
        self.categorical_columns = feature_info['categorical']
        self.continuous_columns = feature_info['continuous']
        
        # Encode categorical features
        train_df, val_df, test_df = self.encode_categorical_features(
            train_df, val_df, test_df, self.categorical_columns
        )
        
        # Standardize continuous features
        train_df, val_df, test_df = self.standardize_continuous_features(
            train_df, val_df, test_df, self.continuous_columns
        )
        
        # Extract X (features only)
        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION COMPLETE (REGIME CLASSIFICATION)")
        logger.info("="*80)
        logger.info(f"Train: X={X_train.shape}, y_regime={y_train_regime.shape}")
        logger.info(f"Val:   X={X_val.shape}, y_regime={y_val_regime.shape}")
        logger.info(f"Test:  X={X_test.shape}, y_regime={y_test_regime.shape}")
        
        return (
            (X_train, y_train_regime),
            (X_val, y_val_regime),
            (X_test, y_test_regime),
            (y_train_dart, y_val_dart, y_test_dart)
        )


def main():
    """Main training pipeline."""
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Train Regime Classifier for DART spread')
        parser.add_argument('--features_path', type=str, default=None, 
                            help='Path to features directory or parquet file')
        args = parser.parse_args()
        
        logger.info("="*80)
        logger.info("REGIME CLASSIFIER TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        if args.features_path:
            features_path = os.path.join(args.features_path, "hourly_features.parquet")
            logger.info(f"Using command-line features path: {features_path}")
        else:
            features_path = load_features_from_aml_input("features")
        
        # Load and prepare data with regime loader
        loader = RegimeDataLoader(features_path)
        (X_train, y_train_regime), \
        (X_val, y_val_regime), \
        (X_test, y_test_regime), \
        (y_train_dart, y_val_dart, y_test_dart) = loader.prepare_datasets_regime_classification()
        
        # Initialize and train model
        model = RegimeClassifierModel(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            device='gpu',
            num_class=4
        )
        
        model.fit(
            X_train, y_train_regime,
            X_val=X_val, y_val=y_val_regime
        )
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = model.predict(X_train)
        train_metrics = evaluate_classifier(y_train_regime, y_train_pred, "Train")
        log_metrics_to_mlflow({
            'accuracy': train_metrics['accuracy'],
            'f1_macro': train_metrics['f1_macro'],
            'f1_weighted': train_metrics['f1_weighted']
        }, "train")
        
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_classifier(y_val_regime, y_val_pred, "Validation")
        log_metrics_to_mlflow({
            'accuracy': val_metrics['accuracy'],
            'f1_macro': val_metrics['f1_macro'],
            'f1_weighted': val_metrics['f1_weighted']
        }, "val")
        
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_classifier(y_test_regime, y_test_pred, "Test")
        log_metrics_to_mlflow({
            'accuracy': test_metrics['accuracy'],
            'f1_macro': test_metrics['f1_macro'],
            'f1_weighted': test_metrics['f1_weighted']
        }, "test")
        
        # Probability predictions for test set (sample)
        logger.info("\n" + "="*80)
        logger.info("SAMPLE PREDICTIONS WITH PROBABILITIES")
        logger.info("="*80)
        
        y_test_proba = model.predict_proba(X_test)
        
        # Show first 10 predictions
        logger.info("\nFirst 10 test predictions:")
        logger.info(f"{'Actual':<10} {'Predicted':<10} {'Prob R0':<10} {'Prob R1':<10} {'Prob R2':<10} {'Prob R3':<10} {'DART Value':<12}")
        logger.info("-" * 80)
        for i in range(min(10, len(y_test_regime))):
            logger.info(
                f"Regime {y_test_regime[i]:<3} "
                f"Regime {y_test_pred[i]:<3} "
                f"{y_test_proba[i][0]:>8.4f}  "
                f"{y_test_proba[i][1]:>8.4f}  "
                f"{y_test_proba[i][2]:>8.4f}  "
                f"{y_test_proba[i][3]:>8.4f}  "
                f"${y_test_dart[i]:>10.2f}"
            )
        
        # Feature importance
        logger.info("\n" + "="*80)
        logger.info("TOP 20 FEATURE IMPORTANCES")
        logger.info("="*80)
        
        feature_importance = model.get_feature_importance(importance_type='gain')
        feature_names = loader.feature_columns
        
        importance_df = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        for i, (name, importance) in enumerate(importance_df, 1):
            logger.info(f"  {i:2d}. {name:50s} {importance:10.2f}")
        
        # Save model
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        # Get output path from Azure ML or use local
        output_path = os.environ.get("AZUREML_OUTPUT_model", "outputs")
        os.makedirs(output_path, exist_ok=True)
        
        model_path = os.path.join(output_path, "regime_classifier_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'lgb_model': model.model,
                'feature_columns': loader.feature_columns,
                'scaler': loader.scaler,
                'categorical_encoders': loader.categorical_encoders,
                'regime_definitions': {
                    0: '|DART| < 10',
                    1: '10 <= |DART| < 50',
                    2: '50 <= |DART| < 100',
                    3: '|DART| >= 100'
                },
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }, f)
        
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info("  Saved components:")
        logger.info("    - RegimeClassifierModel (complete model)")
        logger.info("    - LightGBM classifier")
        logger.info("    - Feature preprocessing")
        logger.info("    - Regime definitions")
        logger.info("    - Performance metrics")
        
        logger.info("\n" + "="*80)
        logger.info("REGIME CLASSIFIER TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
        logger.info("\nThis model predicts which magnitude bucket a DART spread will fall into:")
        logger.info("  Regime 0: |DART| < 10        (Low spread)")
        logger.info("  Regime 1: 10 <= |DART| < 50  (Medium spread)")
        logger.info("  Regime 2: 50 <= |DART| < 100 (High spread)")
        logger.info("  Regime 3: |DART| >= 100      (Very high spread)")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

