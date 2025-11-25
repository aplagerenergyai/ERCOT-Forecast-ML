#!/usr/bin/env python3
"""
ERCOT Feature Engineering - Parquet Validation Script

Validates the output parquet file from Step 1 (feature engineering).

Usage:
    python validate_parquet.py [--file PATH]

    If --file is not provided, will attempt to download from Azure ML workspaceblobstore.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_parquet(file_path: str) -> bool:
    """
    Validate the parquet file and return True if all checks pass.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        True if validation passes, False otherwise
    """
    print_header("ERCOT FEATURE ENGINEERING - PARQUET VALIDATION")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📁 File: {file_path}")
    print(f"💾 Size: {file_size_mb:.2f} MB")
    
    # Load parquet file
    print("\n⏳ Loading parquet file...")
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading parquet: {e}")
        return False
    
    # Track validation results
    all_checks_passed = True
    
    # -------------------------------------------------------------------------
    # Check 1: Basic Shape and Memory
    # -------------------------------------------------------------------------
    print_header("1. BASIC INFORMATION")
    
    print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"💾 Memory Usage: {memory_mb:.2f} MB")
    
    if df.shape[0] == 0:
        print("❌ FAIL: DataFrame is empty!")
        all_checks_passed = False
    else:
        print("✅ PASS: DataFrame has data")
    
    # -------------------------------------------------------------------------
    # Check 2: Column Names
    # -------------------------------------------------------------------------
    print_header("2. COLUMNS")
    
    print(f"📋 Total Columns: {len(df.columns)}")
    print("\nColumn List:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"  {i:2d}. {col:40s} ({dtype:10s}) - {non_null:,} non-null ({100-null_pct:.1f}%)")
    
    # Check for required columns
    required_columns = ['TimestampHour']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ FAIL: Missing required columns: {missing_cols}")
        all_checks_passed = False
    else:
        print(f"\n✅ PASS: All required columns present")
    
    # -------------------------------------------------------------------------
    # Check 3: Timestamp Analysis
    # -------------------------------------------------------------------------
    print_header("3. TIMESTAMP ANALYSIS")
    
    if 'TimestampHour' in df.columns:
        df_sorted = df.sort_values('TimestampHour')
        
        min_time = df_sorted['TimestampHour'].min()
        max_time = df_sorted['TimestampHour'].max()
        total_days = (max_time - min_time).days
        
        print(f"📅 Date Range:")
        print(f"   Start: {min_time}")
        print(f"   End:   {max_time}")
        print(f"   Span:  {total_days:,} days")
        
        # Check for time gaps
        time_diffs = df_sorted['TimestampHour'].diff()
        expected_diff = pd.Timedelta(hours=1)
        
        # Count 1-hour intervals
        one_hour_intervals = (time_diffs == expected_diff).sum()
        
        # Find gaps > 1 hour
        gaps = time_diffs[time_diffs > expected_diff]
        
        print(f"\n⏰ Time Continuity:")
        print(f"   Expected 1-hour intervals: {one_hour_intervals:,}")
        print(f"   Time gaps > 1 hour: {len(gaps):,}")
        
        if len(gaps) > 0:
            print(f"\n⚠️  First 10 time gaps:")
            for idx, gap in gaps.head(10).items():
                timestamp = df_sorted.loc[idx, 'TimestampHour']
                print(f"      {timestamp}: gap of {gap}")
            
            if len(gaps) > 100:
                print(f"\n⚠️  WARNING: {len(gaps):,} time gaps found (this may be expected for missing data periods)")
            else:
                print(f"\n✅ PASS: Time gaps are within acceptable range")
        else:
            print(f"\n✅ PASS: No time gaps - continuous hourly data")
        
        # Check for duplicate timestamps
        duplicates = df_sorted['TimestampHour'].duplicated().sum()
        if duplicates > 0:
            print(f"\n❌ FAIL: Found {duplicates:,} duplicate timestamps!")
            all_checks_passed = False
        else:
            print(f"✅ PASS: No duplicate timestamps")
    
    # -------------------------------------------------------------------------
    # Check 4: Null Value Analysis
    # -------------------------------------------------------------------------
    print_header("4. NULL VALUE ANALYSIS")
    
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(null_cols) == 0:
        print("✅ No null values found in any column")
    else:
        print(f"⚠️  Columns with null values ({len(null_cols)} total):\n")
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            print(f"   {col:40s}: {count:,} ({pct:.1f}%)")
        
        # Warn if any column is >50% null
        high_null_cols = null_cols[null_cols / len(df) > 0.5]
        if len(high_null_cols) > 0:
            print(f"\n⚠️  WARNING: {len(high_null_cols)} columns are >50% null:")
            for col in high_null_cols.index:
                pct = (null_cols[col] / len(df)) * 100
                print(f"      {col}: {pct:.1f}% null")
    
    # -------------------------------------------------------------------------
    # Check 5: DART Spread Calculation
    # -------------------------------------------------------------------------
    print_header("5. DART SPREAD VALIDATION")
    
    if 'DAM_Price_Hourly' in df.columns and 'RTM_LMP_HourlyAvg' in df.columns:
        dam_count = df['DAM_Price_Hourly'].notna().sum()
        rtm_count = df['RTM_LMP_HourlyAvg'].notna().sum()
        
        print(f"📊 Price Data Availability:")
        print(f"   DAM prices present: {dam_count:,} rows ({dam_count/len(df)*100:.1f}%)")
        print(f"   RTM prices present: {rtm_count:,} rows ({rtm_count/len(df)*100:.1f}%)")
        
        # Calculate DART spread
        df['_temp_dart'] = df['DAM_Price_Hourly'] - df['RTM_LMP_HourlyAvg']
        valid_dart = df['_temp_dart'].notna().sum()
        
        if valid_dart > 0:
            print(f"\n📈 DART Spread Statistics:")
            print(f"   Valid DART spread rows: {valid_dart:,} ({valid_dart/len(df)*100:.1f}%)")
            print(f"   Mean: ${df['_temp_dart'].mean():.2f}")
            print(f"   Std:  ${df['_temp_dart'].std():.2f}")
            print(f"   Min:  ${df['_temp_dart'].min():.2f}")
            print(f"   25%:  ${df['_temp_dart'].quantile(0.25):.2f}")
            print(f"   50%:  ${df['_temp_dart'].quantile(0.50):.2f}")
            print(f"   75%:  ${df['_temp_dart'].quantile(0.75):.2f}")
            print(f"   Max:  ${df['_temp_dart'].max():.2f}")
            
            if valid_dart < len(df) * 0.5:
                print(f"\n⚠️  WARNING: DART spread only available for {valid_dart/len(df)*100:.1f}% of rows")
                print("   This may impact model training quality")
            else:
                print(f"\n✅ PASS: DART spread available for {valid_dart/len(df)*100:.1f}% of rows")
        else:
            print(f"\n❌ FAIL: No valid DART spread values found!")
            all_checks_passed = False
        
        # Clean up temp column
        df.drop(columns=['_temp_dart'], inplace=True)
    else:
        print("⚠️  WARNING: DAM and/or RTM price columns not found")
        print("   Cannot calculate DART spread")
        all_checks_passed = False
    
    # -------------------------------------------------------------------------
    # Check 6: Feature Distributions
    # -------------------------------------------------------------------------
    print_header("6. FEATURE DISTRIBUTIONS")
    
    # Numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"📊 Numeric Columns: {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        print("\nBasic Statistics (first 5 numeric columns):")
        print(df[numeric_cols[:5]].describe())
    
    # -------------------------------------------------------------------------
    # Check 7: Settlement Point Analysis (if present)
    # -------------------------------------------------------------------------
    print_header("7. SETTLEMENT POINT ANALYSIS")
    
    if 'SettlementPoint' in df.columns:
        settlement_counts = df['SettlementPoint'].value_counts()
        print(f"📍 Unique Settlement Points: {len(settlement_counts):,}")
        print(f"\nTop 10 Settlement Points by Row Count:")
        for i, (sp, count) in enumerate(settlement_counts.head(10).items(), 1):
            print(f"   {i:2d}. {sp:40s}: {count:,} rows")
    else:
        print("ℹ️  No SettlementPoint column found (this may be expected for aggregated data)")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print_header("VALIDATION SUMMARY")
    
    if all_checks_passed:
        print("✅ ✅ ✅  ALL VALIDATION CHECKS PASSED  ✅ ✅ ✅")
        print("\n🎉 The parquet file is ready for model training!")
        print("\nNext steps:")
        print("  1. Run: make train")
        print("  2. Or run: ./run_training_jobs.sh")
        return True
    else:
        print("❌ ❌ ❌  VALIDATION FAILED  ❌ ❌ ❌")
        print("\n⚠️  The parquet file has issues that need to be addressed.")
        print("   Review the errors above and fix the feature engineering pipeline.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ERCOT feature engineering parquet output"
    )
    parser.add_argument(
        '--file',
        '-f',
        type=str,
        default=None,
        help='Path to parquet file (default: auto-detect from local output or Azure ML)'
    )
    
    args = parser.parse_args()
    
    # Determine file path
    if args.file:
        file_path = args.file
    else:
        # Try to find parquet file in common locations
        possible_paths = [
            'data/features/hourly_features.parquet',
            'features_output/features/hourly_features.parquet',
            'job_output/features/hourly_features.parquet',
            './hourly_features.parquet',
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"ℹ️  Auto-detected parquet file: {path}")
                break
        
        if not file_path:
            print("❌ ERROR: Could not find parquet file")
            print("\nPlease specify file path with --file option, or download from Azure ML:")
            print("  az ml job download --name <JOB_NAME> --workspace-name energyaiml-prod \\")
            print("    --resource-group rg-ercot-ml-production --download-path ./features_output \\")
            print("    --output-name features")
            sys.exit(1)
    
    # Run validation
    success = validate_parquet(file_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

