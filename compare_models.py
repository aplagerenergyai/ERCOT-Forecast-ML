#!/usr/bin/env python3
"""
compare_models.py

Automatically compare all trained DART models and identify the best one.
"""

import subprocess
import json
import re
from typing import Dict, List, Tuple
from datetime import datetime


def run_az_command(cmd: List[str]) -> str:
    """Run Azure CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return ""


def get_completed_jobs() -> List[Dict]:
    """Get all completed DART model jobs."""
    print("  Querying Azure ML for completed jobs...")
    cmd = [
        'az', 'ml', 'job', 'list',
        '--workspace-name', 'energyaiml-prod',
        '--resource-group', 'rg-ercot-ml-production',
        '--max-results', '50',
        '--output', 'json'
    ]
    
    output = run_az_command(cmd)
    if not output:
        return []
    
    jobs = json.loads(output)
    
    # Filter for completed DART jobs
    dart_jobs = [
        job for job in jobs
        if job.get('status') == 'Completed' and 'DART' in job.get('display_name', '')
    ]
    
    return dart_jobs


def print_comparison_table(results: List[Tuple[str, str, Dict[str, float]]]):
    """Print formatted comparison table for regression models."""
    # Separate regression and classification models
    regression_results = []
    classification_results = []
    
    for display_name, job_name, metrics in results:
        # Check if it's a classification model (has 'accuracy' but not 'mae')
        if 'accuracy' in metrics and 'mae' not in metrics:
            classification_results.append((display_name, job_name, metrics))
        # Check if it's a regression model (has 'mae')
        elif 'mae' in metrics:
            regression_results.append((display_name, job_name, metrics))
    
    # ========================================================================
    # REGRESSION MODELS TABLE
    # ========================================================================
    if regression_results:
        print("\n" + "="*120)
        print("  🏆 ERCOT DART REGRESSION MODELS - POINT PREDICTION")
        print("="*120)
        print(f"\n{'Model':<40} {'Test MAE':<15} {'Test RMSE':<15} {'Test R²':<12} {'Extra Info':<25} {'Status':<10}")
        print("-"*120)
        
        # Sort by MAE (best first)
        sorted_results = sorted(
            regression_results, 
            key=lambda x: x[2]['mae'] if x[2]['mae'] is not None else float('inf')
        )
        
        for display_name, job_name, metrics in sorted_results:
            mae = f"${metrics['mae']:.2f}" if metrics['mae'] is not None else "Check logs"
            rmse = f"${metrics['rmse']:.2f}" if metrics['rmse'] is not None else "Check logs"
            r2 = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "Check logs"
            
            # Extra info for special models
            extra_info = ""
            if 'coverage_80' in metrics and metrics['coverage_80'] is not None:
                # Quantile model
                extra_info = f"Coverage: {metrics['coverage_80']*100:.1f}%"
            elif 'model_type' in metrics:
                extra_info = metrics['model_type']
            
            # Add emoji for top 3
            rank_emoji = ""
            if metrics['mae'] is not None:
                idx = sorted_results.index((display_name, job_name, metrics))
                if idx == 0:
                    rank_emoji = "🥇"
                elif idx == 1:
                    rank_emoji = "🥈"
                elif idx == 2:
                    rank_emoji = "🥉"
            
            print(f"{display_name:<40} {mae:<15} {rmse:<15} {r2:<12} {extra_info:<25} {'✅'} {rank_emoji}")
        
        print("-"*120)
    
    # ========================================================================
    # CLASSIFICATION MODELS TABLE
    # ========================================================================
    if classification_results:
        print("\n" + "="*100)
        print("  🎯 ERCOT DART CLASSIFICATION MODELS - REGIME PREDICTION")
        print("="*100)
        print(f"\n{'Model':<40} {'Test Accuracy':<20} {'Test F1 (Macro)':<20} {'Status':<10}")
        print("-"*100)
        
        for display_name, job_name, metrics in classification_results:
            accuracy = f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)" if metrics.get('accuracy') is not None else "Check logs"
            f1_macro = f"{metrics['f1_macro']:.4f}" if metrics.get('f1_macro') is not None else "Check logs"
            
            print(f"{display_name:<40} {accuracy:<20} {f1_macro:<20} {'✅'}")
        
        print("-"*100)
        print("\nNote: Classification models predict regime (magnitude bucket), not exact DART value.")
        print("      They are not directly comparable to regression models.")


def find_best_model(results: List[Tuple[str, str, Dict[str, float]]]) -> Tuple[str, Dict[str, float]]:
    """Find the regression model with lowest MAE."""
    # Only consider regression models (have 'mae')
    valid_results = [
        (name, metrics) for name, _, metrics in results 
        if 'mae' in metrics and metrics['mae'] is not None
    ]
    
    if not valid_results:
        return None, None
    
    best = min(valid_results, key=lambda x: x[1]['mae'])
    return best[0], best[1]


def main():
    print("\n" + "="*100)
    print("  🔍 ERCOT ML MODEL COMPARISON TOOL")
    print("="*100)
    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workspace: energyaiml-prod")
    print(f"  Resource Group: rg-ercot-ml-production\n")
    
    jobs = get_completed_jobs()
    
    if not jobs:
        print("❌ No completed DART jobs found.")
        print("\nMake sure you have jobs with 'DART' in the display name that have completed.")
        print("Check: https://ml.azure.com\n")
        return
    
    print(f"✅ Found {len(jobs)} completed DART jobs\n")
    
    results = []
    
    # ============================================================================
    # MANUAL RESULTS - Update these with actual values from Azure ML logs
    # ============================================================================
    # To get these values:
    # 1. Go to https://ml.azure.com
    # 2. Click on each completed job
    # 3. Go to "Outputs + logs" tab
    # 4. Open "user_logs/std_log.txt"
    # 5. Search for "Test Set Metrics:"
    # 6. Copy the MAE, RMSE, R², and MAPE values here
    # ============================================================================
    
    manual_results = {
        # ========================================================================
        # REGRESSION MODELS - Predict exact DART spread value
        # ========================================================================
        # Format: {'mae': float, 'rmse': float, 'r2': float, 'mape': float}
        
        # Current known results:
        'Train_LightGBM_DART_Spread': {
            'mae': 11.90, 'rmse': 124.05, 'r2': 0.0001, 'mape': 291.45
        },
        'Train_XGBoost_DART_Spread': {
            'mae': 12.83, 'rmse': 125.20, 'r2': -0.0015, 'mape': 295.30
        },
        
        # Add these when they complete:
        # 'Train_CatBoost_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_DeepLearning_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_RandomForest_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_ExtraTrees_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_HistGradientBoosting_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_AutoML_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        # 'Train_Ensemble_DART_Spread': {
        #     'mae': None, 'rmse': None, 'r2': None, 'mape': None
        # },
        
        # ========================================================================
        # TWO-STAGE MODEL - Predicts DA and RT separately, then computes DART
        # ========================================================================
        # Use DART predictions (RT_pred - DA_pred) and compute MAE/RMSE
        # Format: Same as regression models
        # 'Train_TwoStage_DART_Model': {
        #     'mae': None,  # From DART spread predictions
        #     'rmse': None,  # From DART spread predictions
        #     'r2': None,
        #     'mape': None,
        #     'model_type': 'Two-Stage'
        # },
        
        # ========================================================================
        # QUANTILE MODEL - Predicts P10, P50, P90
        # ========================================================================
        # Use P50 (median) for MAE/RMSE, add coverage_80 for calibration
        # Format: {'mae': float (P50), 'rmse': float (P50), 'r2': float (P50), 'coverage_80': float}
        # 'Train_Quantile_DART_Model': {
        #     'mae': None,  # MAE of P50 (median) predictions
        #     'rmse': None,  # RMSE of P50 (median) predictions
        #     'r2': None,  # R² of P50 (median) predictions
        #     'mape': None,
        #     'coverage_80': None,  # Coverage of P10-P90 interval (target: 0.80)
        #     'model_type': 'Quantile'
        # },
        
        # ========================================================================
        # CLASSIFICATION MODELS - Predict regime (magnitude bucket)
        # ========================================================================
        # Format: {'accuracy': float, 'f1_macro': float}
        # 'Train_Regime_Classifier': {
        #     'accuracy': None,  # Overall accuracy (target: >0.80)
        #     'f1_macro': None,  # F1 score macro average (target: >0.75)
        # },
    }
    
    print("📊 Collecting results from completed jobs...\n")
    
    for job in jobs:
        display_name = job['display_name']
        job_name = job['name']
        
        # Check if we have manual results
        if display_name in manual_results:
            metrics = manual_results[display_name]
            
            # Different output based on model type
            if 'accuracy' in metrics:
                # Classification model
                if metrics['accuracy'] is not None:
                    print(f"  ✅ {display_name}: Accuracy={metrics['accuracy']*100:.2f}%")
                else:
                    print(f"  ⏳ {display_name}: Waiting for manual entry")
                    print(f"      Job: {job_name}")
                    print(f"      Check logs at: https://ml.azure.com")
            elif 'mae' in metrics:
                # Regression model (including Two-Stage and Quantile)
                if metrics['mae'] is not None:
                    if 'coverage_80' in metrics and metrics['coverage_80'] is not None:
                        print(f"  ✅ {display_name}: MAE=${metrics['mae']:.2f} (P50), Coverage={metrics['coverage_80']*100:.1f}%")
                    else:
                        print(f"  ✅ {display_name}: MAE=${metrics['mae']:.2f}")
                else:
                    print(f"  ⏳ {display_name}: Waiting for manual entry")
                    print(f"      Job: {job_name}")
                    print(f"      Check logs at: https://ml.azure.com")
            else:
                print(f"  ⚠️  {display_name}: Invalid metrics format")
                metrics = {'mae': None, 'rmse': None, 'r2': None, 'mape': None}
        else:
            print(f"  ⚠️  {display_name}: Not in manual_results dict")
            print(f"      Job: {job_name}")
            print(f"      Add to manual_results in this script")
            metrics = {'mae': None, 'rmse': None, 'r2': None, 'mape': None}
        
        results.append((display_name, job_name, metrics))
    
    # Print comparison table
    print_comparison_table(results)
    
    # Find and announce best model
    best_name, best_metrics = find_best_model(results)
    
    if best_name and best_metrics:
        print("\n" + "="*100)
        print("  🏆 WINNER - BEST MODEL")
        print("="*100)
        print(f"\n  Model: {best_name}")
        print(f"  Test MAE:  ${best_metrics['mae']:.2f}  ⭐ LOWEST ERROR")
        print(f"  Test RMSE: ${best_metrics['rmse']:.2f}")
        print(f"  Test R²:   {best_metrics['r2']:.4f}")
        print(f"  Test MAPE: {best_metrics['mape']:.2f}%")
        print("\n" + "="*100)
        
        # Calculate improvement over baseline
        baseline = 11.90  # LightGBM baseline
        if best_metrics['mae'] < baseline:
            improvement = ((baseline - best_metrics['mae']) / baseline) * 100
            print(f"\n  🎉 {improvement:.1f}% improvement over LightGBM baseline (${baseline:.2f})")
            print(f"  💰 Average error reduced by ${baseline - best_metrics['mae']:.2f} per prediction")
        elif best_metrics['mae'] == baseline:
            print(f"\n  ℹ️  Matches LightGBM baseline performance (${baseline:.2f})")
        else:
            decline = ((best_metrics['mae'] - baseline) / baseline) * 100
            print(f"\n  ⚠️  {decline:.1f}% worse than LightGBM baseline (${baseline:.2f})")
        
        print()
    else:
        print("\n" + "="*100)
        print("  ⚠️  NO VALID METRICS FOUND")
        print("="*100)
        print("\n  To add results:")
        print("  1. Go to https://ml.azure.com")
        print("  2. Click each completed job")
        print("  3. Navigate to 'Outputs + logs' → 'user_logs/std_log.txt'")
        print("  4. Search for 'Test Set Metrics:'")
        print("  5. Update the 'manual_results' dictionary in this script")
        print(f"  6. Rerun: python {__file__}\n")
    
    # Print summary stats for regression models
    valid_regression = [r for r in results if 'mae' in r[2] and r[2]['mae'] is not None]
    valid_classification = [r for r in results if 'accuracy' in r[2] and r[2]['accuracy'] is not None]
    
    if len(valid_regression) > 1:
        print("\n" + "="*100)
        print("  📊 REGRESSION MODELS - SUMMARY STATISTICS")
        print("="*100)
        maes = [r[2]['mae'] for r in valid_regression]
        print(f"\n  Models evaluated: {len(valid_regression)}")
        print(f"  Best MAE:   ${min(maes):.2f}")
        print(f"  Worst MAE:  ${max(maes):.2f}")
        print(f"  Average MAE: ${sum(maes)/len(maes):.2f}")
        print(f"  Range:      ${max(maes) - min(maes):.2f}")
        print()
    
    if len(valid_classification) > 0:
        print("\n" + "="*100)
        print("  📊 CLASSIFICATION MODELS - SUMMARY")
        print("="*100)
        for display_name, _, metrics in valid_classification:
            print(f"\n  {display_name}:")
            print(f"    Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"    F1 (Macro):  {metrics['f1_macro']:.4f}")
        print()


if __name__ == "__main__":
    main()

