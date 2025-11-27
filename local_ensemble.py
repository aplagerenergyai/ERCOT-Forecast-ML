"""
local_ensemble.py

Download trained models from Azure ML and create a weighted ensemble locally.
This script provides 80% of the benefit of a full ensemble with minimal infrastructure.

Usage:
    python local_ensemble.py --features data/hourly_features.parquet --output predictions.csv
"""

import os
import sys
import logging
import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloader import ERCOTDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Job IDs of successfully trained models
MODEL_JOBS = {
    'lgbm': 'silver_egg_8qnwzpj2sl',
    'xgb': 'mighty_ear_jlrxzxzt7g',
    'catboost': 'maroon_camera_897lwtvxgh',
    'rf': 'patient_jelly_ygcvz566k7',
    'deep': 'stoic_board_tmmrlwm6gb',
    'histgb': 'jolly_cheetah_360cptc56l',
    'extratrees': 'helpful_arch_997czfrfj6',
    'tabnet': 'loving_horse_mbmxvfdx6v',
    'automl': 'blue_yam_v1t1rnnhh0',
}


def download_model(model_name, job_id, output_dir='./models'):
    """
    Download model artifacts from Azure ML job.
    
    Args:
        model_name: Name of the model (e.g., 'lgbm')
        job_id: Azure ML job ID
        output_dir: Local directory to save models
    
    Returns:
        Path to downloaded model file, or None if failed
    """
    logger.info(f"📥 Downloading {model_name} model from job {job_id}...")
    
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use az CLI to download
        cmd = [
            'az', 'ml', 'job', 'download',
            '--name', job_id,
            '--output-name', 'model',
            '--download-path', str(model_dir),
            '--resource-group', 'rg-ercot-ml-production',
            '--workspace-name', 'energyaiml-prod'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Find the model file
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pkl') or file.endswith('.pt'):
                        model_path = os.path.join(root, file)
                        logger.info(f"  ✓ Found: {file}")
                        return model_path
            
            logger.warning(f"  ✗ No model file found in {model_dir}")
            return None
        else:
            logger.error(f"  ✗ Download failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"  ✗ Error downloading {model_name}: {e}")
        return None


def load_model(model_path, model_name):
    """Load a pickled model."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model from the saved dictionary
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data['model'], model_data.get('metrics', {})
        else:
            return model_data, {}
            
    except Exception as e:
        logger.error(f"  ✗ Failed to load {model_name}: {e}")
        return None, {}


def generate_predictions(model, model_name, X_val, X_test):
    """
    Generate predictions from a model.
    
    Returns:
        (val_preds, test_preds) or (None, None) if failed
    """
    try:
        logger.info(f"  Predicting with {model_name}...")
        
        # Handle PyTorch models (deep learning)
        if model_name == 'deep' and hasattr(model, 'eval'):
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val)
                X_test_tensor = torch.FloatTensor(X_test)
                val_preds = model(X_val_tensor).numpy().flatten()
                test_preds = model(X_test_tensor).numpy().flatten()
        
        # Handle TabNet
        elif model_name == 'tabnet':
            val_preds = model.predict(X_val).flatten()
            test_preds = model.predict(X_test).flatten()
        
        # Standard sklearn models
        else:
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)
            
            # Ensure flat arrays
            if isinstance(val_preds, np.ndarray) and val_preds.ndim > 1:
                val_preds = val_preds.flatten()
            if isinstance(test_preds, np.ndarray) and test_preds.ndim > 1:
                test_preds = test_preds.flatten()
        
        return val_preds, test_preds
        
    except Exception as e:
        logger.error(f"  ✗ Prediction failed for {model_name}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Create local weighted ensemble from Azure ML models')
    parser.add_argument('--features', type=str, required=True, help='Path to hourly_features.parquet')
    parser.add_argument('--output', type=str, default='ensemble_predictions.csv', help='Output CSV file')
    parser.add_argument('--models-dir', type=str, default='./downloaded_models', help='Directory to save models')
    parser.add_argument('--skip-download', action='store_true', help='Skip download if models already exist')
    parser.add_argument('--max-samples', type=int, default=2_000_000, help='Max samples to use (for memory)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("LOCAL WEIGHTED ENSEMBLE")
    logger.info("="*80)
    
    # Step 1: Download models
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DOWNLOADING MODELS FROM AZURE ML")
    logger.info("="*80)
    
    model_paths = {}
    
    if args.skip_download:
        logger.info("⏭️  Skipping download (using existing models)")
        # Find existing model files
        models_dir = Path(args.models_dir)
        if models_dir.exists():
            for model_name in MODEL_JOBS.keys():
                model_dir = models_dir / model_name
                if model_dir.exists():
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            if file.endswith('.pkl') or file.endswith('.pt'):
                                model_paths[model_name] = os.path.join(root, file)
                                logger.info(f"  ✓ Found existing: {model_name}")
                                break
    else:
        for model_name, job_id in MODEL_JOBS.items():
            model_path = download_model(model_name, job_id, args.models_dir)
            if model_path:
                model_paths[model_name] = model_path
    
    if len(model_paths) < 2:
        logger.error(f"❌ Need at least 2 models for ensemble, found {len(model_paths)}")
        sys.exit(1)
    
    logger.info(f"\n✓ Successfully downloaded/found {len(model_paths)} models")
    
    # Step 2: Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING AND PREPARING DATA")
    logger.info("="*80)
    
    loader = ERCOTDataLoader(args.features)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets(
        max_total_samples=args.max_samples
    )
    
    # Step 3: Load models and generate predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 3: LOADING MODELS AND GENERATING PREDICTIONS")
    logger.info("="*80)
    
    val_predictions = {}
    test_predictions = {}
    val_rmses = {}
    
    for model_name, model_path in model_paths.items():
        logger.info(f"\n📊 Processing {model_name}...")
        
        # Load model
        model, metrics = load_model(model_path, model_name)
        if model is None:
            continue
        
        # Generate predictions
        val_preds, test_preds = generate_predictions(model, model_name, X_val, X_test)
        if val_preds is None:
            continue
        
        val_predictions[model_name] = val_preds
        test_predictions[model_name] = test_preds
        
        # Calculate validation RMSE
        rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
        val_rmses[model_name] = rmse
        logger.info(f"  Validation RMSE: ${rmse:.2f}")
    
    if len(val_predictions) < 2:
        logger.error(f"❌ Need at least 2 models with predictions, got {len(val_predictions)}")
        sys.exit(1)
    
    # Step 4: Calculate weights (inverse RMSE)
    logger.info("\n" + "="*80)
    logger.info("STEP 4: CALCULATING ENSEMBLE WEIGHTS")
    logger.info("="*80)
    
    # Inverse RMSE weighting
    inverse_rmse = {name: 1.0 / rmse for name, rmse in val_rmses.items()}
    total_weight = sum(inverse_rmse.values())
    weights = {name: w / total_weight for name, w in inverse_rmse.items()}
    
    logger.info("\nModel weights (based on validation RMSE):")
    for name in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
        logger.info(f"  {name:15s}: {weights[name]:.3f} (RMSE: ${val_rmses[name]:.2f})")
    
    # Step 5: Create weighted ensemble predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 5: CREATING WEIGHTED ENSEMBLE")
    logger.info("="*80)
    
    # Validation ensemble
    ensemble_val = np.zeros_like(y_val, dtype=float)
    for name, preds in val_predictions.items():
        ensemble_val += preds * weights[name]
    
    # Test ensemble
    ensemble_test = np.zeros_like(y_test, dtype=float)
    for name, preds in test_predictions.items():
        ensemble_test += preds * weights[name]
    
    # Calculate ensemble RMSE
    ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_val) ** 2))
    best_single_rmse = min(val_rmses.values())
    improvement = ((best_single_rmse - ensemble_rmse) / best_single_rmse) * 100
    
    logger.info(f"\n📈 RESULTS:")
    logger.info(f"  Best single model RMSE: ${best_single_rmse:.2f}")
    logger.info(f"  Ensemble RMSE:          ${ensemble_rmse:.2f}")
    logger.info(f"  Improvement:            {improvement:.2f}%")
    
    # Step 6: Save predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVING PREDICTIONS")
    logger.info("="*80)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'actual': y_test,
        'ensemble_prediction': ensemble_test,
        'ensemble_val_rmse': ensemble_rmse,
    })
    
    # Add individual model predictions
    for name, preds in test_predictions.items():
        output_df[f'{name}_prediction'] = preds
        output_df[f'{name}_val_rmse'] = val_rmses[name]
    
    output_df.to_csv(args.output, index=False)
    logger.info(f"✓ Saved predictions to: {args.output}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ENSEMBLE COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

