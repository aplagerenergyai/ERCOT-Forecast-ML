"""
build_features_mock.py

Creates mock ERCOT data for testing the pipeline end-to-end.
Generates synthetic hourly features WITHOUT needing SQL Server.
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_features(days_back: int = 90) -> pd.DataFrame:
    """
    Generate synthetic ERCOT-like features for testing.
    """
    logger.info(f"Generating mock data for last {days_back} days...")
    
    # Create hourly timestamps
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_back)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    n_hours = len(timestamps)
    logger.info(f"Creating {n_hours:,} hourly records...")
    
    # Base features with realistic patterns
    df = pd.DataFrame({
        'TimestampHour': timestamps,
        'SettlementPoint': 'HB_HOUSTON',  # Primary settlement point
        
        # Load features (MW) - daily pattern
        'SystemLoad_ForecastZone_Total': 40000 + 15000 * np.sin(2 * np.pi * np.arange(n_hours) / 24),
        'SystemLoad_WeatherZone_Total': 39500 + 14500 * np.sin(2 * np.pi * np.arange(n_hours) / 24),
        
        # Price features ($/MWh)
        'DAM_Price_Hourly': 30 + 20 * np.random.randn(n_hours) + 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24),
        'RTM_LMP_HourlyAvg': 32 + 22 * np.random.randn(n_hours) + 12 * np.sin(2 * np.pi * np.arange(n_hours) / 24),
        
        # Solar features (MW)
        'Solar_Actual_5Min_HourlyAvg': np.maximum(0, 3000 * np.sin(np.pi * (np.arange(n_hours) % 24) / 12)),
        'Solar_Forecast_Hourly': np.maximum(0, 3100 * np.sin(np.pi * (np.arange(n_hours) % 24) / 12)),
        
        # Wind features (MW)
        'Wind_Hourly_Actual_Total': 8000 + 4000 * np.random.randn(n_hours),
        'Wind_Hourly_Forecast_Total': 8200 + 3800 * np.random.randn(n_hours),
        
        # Constraint features ($/MWh)
        'SCED_ShadowPrice_5Min_Max': 50 + 30 * np.random.rand(n_hours),
        'SCED_ShadowPrice_5Min_Avg': 25 + 15 * np.random.rand(n_hours),
    })
    
    # Clip prices to realistic ranges
    df['DAM_Price_Hourly'] = df['DAM_Price_Hourly'].clip(0, 150)
    df['RTM_LMP_HourlyAvg'] = df['RTM_LMP_HourlyAvg'].clip(0, 150)
    df['Wind_Hourly_Actual_Total'] = df['Wind_Hourly_Actual_Total'].clip(0, 20000)
    df['Wind_Hourly_Forecast_Total'] = df['Wind_Hourly_Forecast_Total'].clip(0, 20000)
    
    # Add temporal features
    df['Hour'] = df['TimestampHour'].dt.hour
    df['DayOfWeek'] = df['TimestampHour'].dt.dayofweek
    df['Month'] = df['TimestampHour'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Calculate DART spread (target variable)
    df['DART_Spread'] = df['DAM_Price_Hourly'] - df['RTM_LMP_HourlyAvg']
    
    logger.info(f"✅ Generated {len(df):,} hourly records")
    logger.info(f"   Date range: {df['TimestampHour'].min()} to {df['TimestampHour'].max()}")
    logger.info(f"   Features: {len(df.columns)} columns")
    
    return df


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("MOCK FEATURE BUILDER - Testing Mode")
    logger.info("=" * 60)
    
    # Generate mock data
    df = generate_mock_features(days_back=90)
    
    # Get output path from Azure ML environment variable
    output_path = os.environ.get("AZUREML_OUTPUT_features", "./outputs/features")
    os.makedirs(output_path, exist_ok=True)
    
    # Save to parquet
    output_file = os.path.join(output_path, "hourly_features.parquet")
    df.to_parquet(output_file, index=False)
    
    logger.info(f"✅ Saved features to: {output_file}")
    logger.info(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    logger.info("=" * 60)
    logger.info("SUCCESS! Mock features created.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

