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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from dataloader import load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_tft(df_train, df_val, df_test, feature_columns):
    """
    Train Temporal Fusion Transformer.
    
    Args:
        df_train: Training dataframe with TimestampHour
        df_val: Validation dataframe
        df_test: Test dataframe
        feature_columns: List of feature column names
    
    Returns:
        Trained TFT model
    """
    logger.info("Preparing TimeSeriesDataSet...")
    
    # Create time index (sequential integer)
    df_train['time_idx'] = (df_train['TimestampHour'] - df_train['TimestampHour'].min()).dt.total_seconds() / 3600
    df_train['time_idx'] = df_train['time_idx'].astype(int)
    
    # Add group ID (for multi-series, using SettlementPoint if available)
    if 'SettlementPoint' in df_train.columns:
        df_train['group_id'] = df_train['SettlementPoint']
    else:
        df_train['group_id'] = 'all'
    
    # Define parameters
    max_prediction_length = 1  # Predict 1 hour ahead
    max_encoder_length = 24  # Use 24 hours of history
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="DART_Spread",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[col for col in feature_columns if col in df_train.columns],
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
    
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        enable_progress_bar=True,
    )
    
    logger.info("Training TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
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
        
        # Load data
        logger.info(f"Loading features from: {features_path}")
        df = pd.read_parquet(features_path)
        logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Create DART target
        logger.info("Creating DART target variable...")
        df['DART_Spread'] = df['DAM_Price_Hourly'] - df['RTM_LMP_HourlyAvg']
        df = df.dropna(subset=['DART_Spread', 'DAM_Price_Hourly', 'RTM_LMP_HourlyAvg'])
        logger.info(f"  Final dataset: {len(df):,} rows")
        
        # Prepare features
        exclude_cols = ['TimestampHour', 'DAM_Price_Hourly', 'RTM_LMP_HourlyAvg', 'DART_Spread']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Time-based split
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.9)
        
        df_train = df.iloc[:n_train].copy()
        df_val = df.iloc[n_train:n_val].copy()
        df_test = df.iloc[n_val:].copy()
        
        logger.info(f"Train: {len(df_train):,} rows")
        logger.info(f"Val:   {len(df_val):,} rows")
        logger.info(f"Test:  {len(df_test):,} rows")
        
        # Train model
        logger.info("="*80)
        logger.info("TRAINING TFT MODEL")
        logger.info("="*80)
        model, trainer = train_tft(df_train, df_val, df_test, feature_cols)
        
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

