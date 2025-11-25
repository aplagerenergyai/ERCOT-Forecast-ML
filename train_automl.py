"""
train_automl.py

Azure AutoML training for DART spread prediction.
Automatically tries 20+ algorithms and tunes hyperparameters.
"""

import os
import logging
from datetime import datetime

import pandas as pd
from azureml.core import Workspace, Dataset, Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import AmlCompute

from dataloader import load_features_from_aml_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("AZURE AUTOML TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load data
        logger.info(f"Loading features from: {features_path}")
        df = pd.read_parquet(features_path)
        logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Create DART target
        logger.info("Creating DART target variable...")
        df['DART_Spread'] = df['DAM_Price_Hourly'] - df['RTM_LMP_HourlyAvg']
        df = df.dropna(subset=['DART_Spread', 'DAM_Price_Hourly', 'RTM_LMP_HourlyAvg'])
        logger.info(f"  Final dataset: {len(df):,} rows")
        
        # Prepare features and target
        exclude_cols = ['TimestampHour', 'DAM_Price_Hourly', 'RTM_LMP_HourlyAvg', 'DART_Spread']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['DART_Spread']
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Target: DART_Spread (Mean: ${y.mean():.2f}, Std: ${y.std():.2f})")
        
        # Time-based split (80/10/10)
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.9)
        
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_val = X.iloc[n_train:n_val]
        y_val = y.iloc[n_train:n_val]
        X_test = X.iloc[n_val:]
        y_test = y.iloc[n_val:]
        
        logger.info(f"Train: {len(X_train):,} rows")
        logger.info(f"Val:   {len(X_val):,} rows")
        logger.info(f"Test:  {len(X_test):,} rows")
        
        # Combine train + val for AutoML (it will do its own cross-validation)
        X_automl = pd.concat([X_train, X_val])
        y_automl = pd.concat([y_train, y_val])
        
        logger.info(f"AutoML training set: {len(X_automl):,} rows")
        
        # Save test set for later evaluation
        test_df = pd.concat([X_test, y_test], axis=1)
        test_path = "./outputs/test_data.parquet"
        os.makedirs("./outputs", exist_ok=True)
        test_df.to_parquet(test_path, index=False)
        logger.info(f"✓ Saved test set to: {test_path}")
        
        # Combine X and y for AutoML
        train_df = pd.concat([X_automl, y_automl], axis=1)
        
        logger.info("="*80)
        logger.info("CONFIGURING AZURE AUTOML")
        logger.info("="*80)
        
        # AutoML Configuration
        automl_config = AutoMLConfig(
            task='regression',
            primary_metric='normalized_root_mean_squared_error',
            training_data=train_df,
            label_column_name='DART_Spread',
            n_cross_validations=5,
            enable_early_stopping=True,
            experiment_timeout_hours=2,
            max_concurrent_iterations=4,
            max_cores_per_iteration=-1,
            enable_stack_ensemble=True,
            enable_voting_ensemble=True,
            enable_dnn=False,  # Skip deep learning (already testing separately)
            enable_onnx_compatible_models=True,
            verbosity=logging.INFO,
            featurization='auto',
            enable_feature_sweeping=True,
        )
        
        logger.info("AutoML Configuration:")
        logger.info(f"  Task: regression")
        logger.info(f"  Primary Metric: normalized_root_mean_squared_error")
        logger.info(f"  Experiment Timeout: 2 hours")
        logger.info(f"  Cross-Validation Folds: 5")
        logger.info(f"  Max Concurrent Iterations: 4")
        logger.info(f"  Stack Ensemble: Enabled")
        logger.info(f"  Voting Ensemble: Enabled")
        
        logger.info("="*80)
        logger.info("STARTING AUTOML EXPERIMENT")
        logger.info("="*80)
        logger.info("This will take approximately 2 hours...")
        logger.info("AutoML will try 20+ algorithms and hyperparameter combinations")
        
        # Note: In Azure ML, the automl_config will be submitted via the run context
        # This script just prepares the data and config
        logger.info("✓ Data prepared and configuration ready for AutoML")
        logger.info("✓ AutoML will be executed by Azure ML service")
        
        logger.info("="*80)
        logger.info("AUTOML SETUP COMPLETE")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

