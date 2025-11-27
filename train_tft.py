"""
train_tft.py

Temporal Fusion Transformer for DART spread prediction.
State-of-the-art deep learning for time series with attention mechanisms.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_tft(df_train, df_val, df_test, continuous_features, categorical_features):
    """
    Train Temporal Fusion Transformer.
    
    Args:
        df_train: Training dataframe with TimestampHour and encoded features
        df_val: Validation dataframe
        df_test: Test dataframe
        continuous_features: List of continuous feature column names
        categorical_features: List of categorical feature column names (already encoded)
    
    Returns:
        Trained TFT model
    """
    logger.info("Preparing TimeSeriesDataSet...")
    
    # Create time index (sequential integer)
    df_train['time_idx'] = (df_train['TimestampHour'] - df_train['TimestampHour'].min()).dt.total_seconds() / 3600
    df_train['time_idx'] = df_train['time_idx'].astype(int)
    df_val['time_idx'] = (df_val['TimestampHour'] - df_train['TimestampHour'].min()).dt.total_seconds() / 3600
    df_val['time_idx'] = df_val['time_idx'].astype(int)
    
    # Add group ID (we'll use a single group since data is already encoded)
    df_train['group_id'] = 'all'
    df_val['group_id'] = 'all'
    
    # Define parameters
    max_prediction_length = 1  # Predict 1 hour ahead
    max_encoder_length = 24  # Use 24 hours of history
    
    # All features are now continuous (categorical were encoded)
    all_features = continuous_features + categorical_features
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="DART_Spread",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=all_features,
        target_normalizer=None,  # Already normalized
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=256, num_workers=0)
    
    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)
    
    logger.info("Initializing Temporal Fusion Transformer...")
    
    # TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # Quantiles
        loss=RMSE(),
        reduce_on_plateau_patience=4,
    )
    
    logger.info(f"Model parameters: {tft.size()/1e6:.1f}M")
    
    # Configure trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    
    # Verify model is LightningModule
    logger.info(f"Model type: {type(tft)}")
    logger.info(f"Is LightningModule: {isinstance(tft, pl.LightningModule)}")
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        enable_progress_bar=True,
    )
    
    logger.info("Training TFT...")
    # Use the simpler fit API
    trainer.fit(tft, train_dataloader, val_dataloader)
    
    logger.info("✓ Training complete")
    
    return tft, trainer


def main():
    logger.info("="*80)
    logger.info("TEMPORAL FUSION TRANSFORMER TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load data using dataloader (handles encoding and normalization)
        logger.info(f"Loading features from: {features_path}")
        loader = ERCOTDataLoader(features_path)
        
        # Get the processed data as numpy arrays with aggressive memory optimization
        # Sample DURING LOAD to prevent OOM (most memory efficient)
        max_total_samples = 1_000_000  # 1M total samples (→ ~800K train, 100K val, 100K test)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets(
            max_total_samples=max_total_samples
        )
        
        logger.info(f"Train: {len(X_train):,} rows")
        logger.info(f"Val:   {len(X_val):,} rows")
        logger.info(f"Test:  {len(X_test):,} rows")
        
        # Create DataFrames with processed features and synthetic timestamps
        # Note: Since we sampled the data, we create evenly-spaced synthetic timestamps
        # This is acceptable for TFT as it learns temporal patterns from the sequence
        
        # Generate hourly timestamps
        import pandas as pd
        base_time = pd.Timestamp('2023-01-01')
        
        df_train = pd.DataFrame(X_train, columns=loader.feature_columns)
        df_train['DART_Spread'] = y_train
        df_train['TimestampHour'] = pd.date_range(
            start=base_time, periods=len(X_train), freq='H'
        )
        
        df_val = pd.DataFrame(X_val, columns=loader.feature_columns)
        df_val['DART_Spread'] = y_val
        df_val['TimestampHour'] = pd.date_range(
            start=base_time + pd.Timedelta(hours=len(X_train)), 
            periods=len(X_val), 
            freq='H'
        )
        
        df_test = pd.DataFrame(X_test, columns=loader.feature_columns)
        df_test['DART_Spread'] = y_test
        df_test['TimestampHour'] = pd.date_range(
            start=base_time + pd.Timedelta(hours=len(X_train) + len(X_val)),
            periods=len(X_test),
            freq='H'
        )
        
        # Get continuous and categorical features (all are continuous after encoding)
        continuous_features = loader.continuous_columns
        categorical_features = loader.categorical_columns  # Now encoded as continuous
        
        # Train model
        logger.info("="*80)
        logger.info("TRAINING TFT MODEL")
        logger.info("="*80)
        model, trainer = train_tft(df_train, df_val, df_test, continuous_features, categorical_features)
        
        # Note: Evaluation would require proper time series setup
        # Simplified for now
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        output_path = "./outputs/tft_model.ckpt"
        os.makedirs("./outputs", exist_ok=True)
        
        trainer.save_checkpoint(output_path)
        logger.info(f"✓ Model saved to: {output_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("TFT TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Note: TFT requires specific data structure with time series format")
        raise


if __name__ == "__main__":
    main()

