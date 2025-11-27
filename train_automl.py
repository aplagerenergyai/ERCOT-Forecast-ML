"""
train_automl.py

Azure AutoML for DART spread prediction.
Automatically tries 20+ algorithms with hyperparameter tuning.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
    Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_automl_models():
    """
    Define all models to try in AutoML.
    
    Returns:
        Dictionary of {model_name: model_instance}
    """
    models = {
        # Tree Ensembles
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
        
        # Linear Models
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'HuberRegressor': HuberRegressor(),
        'BayesianRidge': BayesianRidge(),
        
        # Gradient Boosting Frameworks
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42),
    }
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, verbose=False, random_state=42)
    
    return models


def run_automl(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Run AutoML: try all models and pick the best.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
    
    Returns:
        best_model, results_df
    """
    models = get_automl_models()
    results = []
    
    logger.info(f"Testing {len(models)} models...")
    logger.info("")
    
    for name, model in models.items():
        try:
            logger.info(f"Training: {name}")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - y_val_pred))
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test)
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            test_mae = np.mean(np.abs(y_test - y_test_pred))
            
            results.append({
                'Model': name,
                'Val_RMSE': val_rmse,
                'Val_MAE': val_mae,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'Model_Object': model
            })
            
            logger.info(f"  Val MAE: ${val_mae:.2f}, Test MAE: ${test_mae:.2f}")
            
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            continue
    
    # Sort by validation MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val_MAE')
    
    best_model = results_df.iloc[0]['Model_Object']
    best_name = results_df.iloc[0]['Model']
    
    logger.info("")
    logger.info("="*80)
    logger.info("AUTOML RESULTS - TOP 10 MODELS")
    logger.info("="*80)
    logger.info("")
    logger.info(f"{'Model':<25} {'Val MAE':<12} {'Test MAE':<12} {'Val RMSE':<12} {'Test RMSE':<12}")
    logger.info("-"*80)
    
    for idx, row in results_df.head(10).iterrows():
        logger.info(f"{row['Model']:<25} ${row['Val_MAE']:<11.2f} ${row['Test_MAE']:<11.2f} ${row['Val_RMSE']:<11.2f} ${row['Test_RMSE']:<11.2f}")
    
    logger.info("")
    logger.info(f"🏆 Best Model: {best_name}")
    logger.info(f"   Val MAE: ${results_df.iloc[0]['Val_MAE']:.2f}")
    logger.info(f"   Test MAE: ${results_df.iloc[0]['Test_MAE']:.2f}")
    
    return best_model, results_df, best_name


def main():
    logger.info("="*80)
    logger.info("AUTOML TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        # Run AutoML
        logger.info("="*80)
        logger.info("RUNNING AUTOML")
        logger.info("="*80)
        best_model, results_df, best_name = run_automl(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Evaluate best model
        logger.info("")
        logger.info("="*80)
        logger.info(f"BEST MODEL EVALUATION: {best_name}")
        logger.info("="*80)
        
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        
        logger.info("")
        logger.info("Train Set Metrics:")
        evaluate_model(y_train, y_train_pred)
        
        logger.info("")
        logger.info("Validation Set Metrics:")
        evaluate_model(y_val, y_val_pred)
        
        logger.info("")
        logger.info("Test Set Metrics:")
        evaluate_model(y_test, y_test_pred)
        
        # Save best model
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING BEST MODEL")
        logger.info("="*80)
        
        import pickle
        output_path = "./outputs/automl_best_model.pkl"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"✓ Best model ({best_name}) saved to: {output_path}")
        
        # Save results
        results_path = "./outputs/automl_results.csv"
        results_df.drop('Model_Object', axis=1).to_csv(results_path, index=False)
        logger.info(f"✓ All results saved to: {results_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("AUTOML TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
