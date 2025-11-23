"""
build_features.py

Loads ERCOT historical data from SQL Server, applies feature engineering,
and produces a unified hourly feature set for ML training.

This script processes 9 ERCOT tables:
- 2 load tables (forecast zone, weather zone) - wide→long
- 2 price tables (DAM, RTM LMP) - hourly and 5-min→hourly
- 1 constraint table (SCED shadow prices) - 5-min→hourly with max/avg
- 2 solar tables (5-min actual, hourly forecast) - resample and merge
- 1 wind table (hourly with regional breakdown) - wide→long for regions

Output: hourly_features.parquet
  - TimestampHour (datetime)
  - SettlementPoint (for price tables)
  - ForecastZone / WeatherZone (for load tables)
  - RegionID (for wind regions)
  - ConstraintName (for transmission constraints)
  - All engineered features with exact Final Names from mapping
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd
import pyodbc
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SQL CONNECTION
# ============================================================================

def get_sql_connection() -> pyodbc.Connection:
    """
    Create SQL Server connection using environment variables.
    Reads from .env file: SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD
    """
    load_dotenv()
    
    server = os.getenv('SQL_SERVER')
    database = os.getenv('SQL_DATABASE')
    username = os.getenv('SQL_USERNAME')
    password = os.getenv('SQL_PASSWORD')
    
    if not all([server, database, username, password]):
        raise ValueError("Missing SQL Server credentials in .env file")
    
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    
    logger.info(f"Connecting to SQL Server: {server}/{database}")
    conn = pyodbc.connect(conn_str)
    logger.info("✓ Connected successfully")
    
    return conn


# ============================================================================
# TIMESTAMP NORMALIZATION UTILITIES
# ============================================================================

def normalize_operday_hourending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OperDay (date) + HourEnding (1-24) → TimestampHour (datetime).
    HourEnding 1 = midnight-1am, 24 = 11pm-midnight.
    """
    df = df.copy()
    
    # Convert OperDay to datetime if not already
    df['OperDay'] = pd.to_datetime(df['OperDay'])
    
    # HourEnding 1-24 → hour offset 0-23
    # HourEnding 1 means the hour ENDING at 1:00, so it starts at 0:00
    df['TimestampHour'] = df['OperDay'] + pd.to_timedelta(df['HourEnding'] - 1, unit='h')
    
    return df


def normalize_deliverydate_hourending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DeliveryDate (datetime) + HourEnding (1-24) → TimestampHour.
    DeliveryDate is already a datetime but may include time component.
    """
    df = df.copy()
    
    # Ensure DeliveryDate is datetime
    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
    
    # Extract just the date part
    df['DeliveryDateOnly'] = df['DeliveryDate'].dt.normalize()
    
    # Add hour offset (HourEnding 1-24 → 0-23 hours)
    df['TimestampHour'] = df['DeliveryDateOnly'] + pd.to_timedelta(df['HourEnding'] - 1, unit='h')
    
    df = df.drop(columns=['DeliveryDateOnly'])
    
    return df


def normalize_sced_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SCEDTimestamp (5-minute intervals) → TimestampHour.
    Floor to the hour for grouping.
    """
    df = df.copy()
    
    df['SCEDTimestamp'] = pd.to_datetime(df['SCEDTimestamp'])
    df['TimestampHour'] = df['SCEDTimestamp'].dt.floor('H')
    
    return df


def normalize_interval_ending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert INTERVAL_ENDING (5-minute intervals) → TimestampHour.
    Floor to the hour.
    """
    df = df.copy()
    
    df['INTERVAL_ENDING'] = pd.to_datetime(df['INTERVAL_ENDING'])
    df['TimestampHour'] = df['INTERVAL_ENDING'].dt.floor('H')
    
    return df


# ============================================================================
# TABLE PROCESSORS
# ============================================================================

def load_table_chunked(conn: pyodbc.Connection, table_name: str, 
                       chunksize: int = 100000) -> pd.DataFrame:
    """
    Load table in chunks to handle large datasets efficiently.
    """
    logger.info(f"Loading {table_name} (chunked, size={chunksize:,})")
    
    query = f"SELECT * FROM {table_name}"
    
    chunks = []
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Get column names
    columns = [desc[0] for desc in cursor.description]
    
    row_count = 0
    while True:
        rows = cursor.fetchmany(chunksize)
        if not rows:
            break
        
        chunk_df = pd.DataFrame.from_records(rows, columns=columns)
        chunks.append(chunk_df)
        row_count += len(chunk_df)
        
        if row_count % 500000 == 0:
            logger.info(f"  ... loaded {row_count:,} rows")
    
    cursor.close()
    
    if not chunks:
        logger.warning(f"  ⚠ No data found in {table_name}")
        return pd.DataFrame()
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# ----------------------------------------------------------------------------
# 1. hist_ActualSystemLoadbyForecastZone
# ----------------------------------------------------------------------------

def process_load_forecast_zone(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Load by Forecast Zone (hourly, wide→long).
    Columns: NORTH, SOUTH, WEST, HOUSTON, TOTAL
    Output: TimestampHour, ForecastZone, Load_{ZONE}_Hourly
    """
    logger.info("\n[1/9] Processing hist_ActualSystemLoadbyForecastZone")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_ActualSystemLoadbyForecastZone]")
    
    if df.empty:
        return pd.DataFrame()
    
    # Normalize timestamp
    df = normalize_operday_hourending(df)
    
    # Melt wide → long
    value_cols = ['NORTH', 'SOUTH', 'WEST', 'HOUSTON', 'TOTAL']
    existing_value_cols = [col for col in value_cols if col in df.columns]
    
    df_long = df.melt(
        id_vars=['TimestampHour'],
        value_vars=existing_value_cols,
        var_name='ForecastZone',
        value_name='LoadValue'
    )
    
    # Rename to final format: Load_{ZONE}_Hourly
    df_long['Feature'] = 'Load_' + df_long['ForecastZone'] + '_Hourly'
    
    # Pivot to get one column per zone
    df_pivot = df_long.pivot_table(
        index='TimestampHour',
        columns='Feature',
        values='LoadValue',
        aggfunc='first'
    ).reset_index()
    
    logger.info(f"  ✓ Output: {len(df_pivot)} rows, {len(df_pivot.columns)-1} features")
    
    return df_pivot


# ----------------------------------------------------------------------------
# 2. hist_ActualSystemLoadbyWeatherZone
# ----------------------------------------------------------------------------

def process_load_weather_zone(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Load by Weather Zone (hourly, wide→long).
    Columns: COAST, EAST, FAR_WEST, NORTH, NORTH_C, SOUTHERN, SOUTH_C, WEST
    """
    logger.info("\n[2/9] Processing hist_ActualSystemLoadbyWeatherZone")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_ActualSystemLoadbyWeatherZone]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_operday_hourending(df)
    
    value_cols = ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTHERN', 'SOUTH_C', 'WEST']
    existing_value_cols = [col for col in value_cols if col in df.columns]
    
    df_long = df.melt(
        id_vars=['TimestampHour'],
        value_vars=existing_value_cols,
        var_name='WeatherZone',
        value_name='LoadValue'
    )
    
    df_long['Feature'] = 'Load_' + df_long['WeatherZone'] + '_Hourly'
    
    df_pivot = df_long.pivot_table(
        index='TimestampHour',
        columns='Feature',
        values='LoadValue',
        aggfunc='first'
    ).reset_index()
    
    logger.info(f"  ✓ Output: {len(df_pivot)} rows, {len(df_pivot.columns)-1} features")
    
    return df_pivot


# ----------------------------------------------------------------------------
# 3. hist_DAMSettlementPointPrices
# ----------------------------------------------------------------------------

def process_dam_prices(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Day-Ahead Market prices (hourly).
    Output: TimestampHour, SettlementPoint, DAM_Price_Hourly
    """
    logger.info("\n[3/9] Processing hist_DAMSettlementPointPrices")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_DAMSettlementPointPrices]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_deliverydate_hourending(df)
    
    # Rename price column
    df = df.rename(columns={'SettlementPointPrice': 'DAM_Price_Hourly'})
    
    # Keep only necessary columns
    df = df[['TimestampHour', 'SettlementPoint', 'DAM_Price_Hourly']]
    
    logger.info(f"  ✓ Output: {len(df)} rows, {df['SettlementPoint'].nunique()} settlement points")
    
    return df


# ----------------------------------------------------------------------------
# 4. hist_LMPbyResourceNodesLoadZonesandTradingHubs
# ----------------------------------------------------------------------------

def process_rtm_lmp(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Real-time LMP (5-minute → hourly average).
    Output: TimestampHour, SettlementPoint, RTM_LMP_HourlyAvg
    """
    logger.info("\n[4/9] Processing hist_LMPbyResourceNodesLoadZonesandTradingHubs")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_LMPbyResourceNodesLoadZonesandTradingHubs]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_sced_timestamp(df)
    
    # Resample 5-min → hourly (average)
    df_hourly = df.groupby(['TimestampHour', 'SettlementPoint'])['LMP'].mean().reset_index()
    df_hourly = df_hourly.rename(columns={'LMP': 'RTM_LMP_HourlyAvg'})
    
    logger.info(f"  ✓ Output: {len(df_hourly)} rows, {df_hourly['SettlementPoint'].nunique()} settlement points")
    
    return df_hourly


# ----------------------------------------------------------------------------
# 5. hist_RealTimeLMP (redundant but included)
# ----------------------------------------------------------------------------

def process_rtm_lmp_alt(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Alternative RTM LMP feed (5-minute → hourly average).
    Usually redundant with table 4, but included per mapping.
    """
    logger.info("\n[5/9] Processing hist_RealTimeLMP")
    
    try:
        df = load_table_chunked(conn, "[ERCOT].[hist_RealTimeLMP]")
        
        if df.empty:
            return pd.DataFrame()
        
        df = normalize_sced_timestamp(df)
        
        df_hourly = df.groupby(['TimestampHour', 'SettlementPoint'])['LMP'].mean().reset_index()
        df_hourly = df_hourly.rename(columns={'LMP': 'RTM_LMP_HourlyAvg_Alt'})
        
        logger.info(f"  ✓ Output: {len(df_hourly)} rows")
        
        return df_hourly
    
    except Exception as e:
        logger.warning(f"  ⚠ Table not found or error: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# 6. hist_SCEDShadowPricesandBindingTransmissionConstraints
# ----------------------------------------------------------------------------

def process_sced_constraints(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    SCED transmission constraints (5-minute → hourly).
    
    Resampling rules:
    - ShadowPrice: max + avg
    - Limit/Flow (Value): avg
    - ViolatedMW: max
    """
    logger.info("\n[6/9] Processing hist_SCEDShadowPricesandBindingTransmissionConstraints")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_SCEDShadowPricesandBindingTransmissionConstraints]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_sced_timestamp(df)
    
    # Aggregate by hour and constraint
    agg_dict = {
        'ShadowPrice': ['max', 'mean'],
        'Limit': 'mean',
        'Value': 'mean',  # Flow
        'ViolatedMW': 'max'
    }
    
    # Only aggregate columns that exist
    existing_agg = {}
    for col, agg in agg_dict.items():
        if col in df.columns:
            existing_agg[col] = agg
    
    df_hourly = df.groupby(['TimestampHour', 'ConstraintName']).agg(existing_agg).reset_index()
    
    # Flatten multi-level columns
    df_hourly.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in df_hourly.columns
    ]
    
    # Rename to final names
    rename_map = {
        'ShadowPrice_max': 'ShadowPrice_HourlyMax',
        'ShadowPrice_mean': 'ShadowPrice_HourlyAvg',
        'Limit_mean': 'ConstraintLimit_HourlyAvg',
        'Value_mean': 'ConstraintFlow_HourlyAvg',
        'ViolatedMW_max': 'ConstraintViolation_HourlyMax'
    }
    
    df_hourly = df_hourly.rename(columns=rename_map)
    
    logger.info(f"  ✓ Output: {len(df_hourly)} rows, {df_hourly['ConstraintName'].nunique()} constraints")
    
    return df_hourly


# ----------------------------------------------------------------------------
# 7. hist_SolarPowerProductionActual5MinuteAveragedValues
# ----------------------------------------------------------------------------

def process_solar_5min(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Solar 5-minute actual (5-minute → hourly average).
    Columns: SYSTEM_WIDE_GEN, SYSTEM_WIDE_HSL
    Output: Solar_Actual_Hourly, Solar_HSL_Hourly
    """
    logger.info("\n[7/9] Processing hist_SolarPowerProductionActual5MinuteAveragedValues")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_SolarPowerProductionActual5MinuteAveragedValues]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_interval_ending(df)
    
    # Resample to hourly (average)
    agg_dict = {}
    if 'SYSTEM_WIDE_GEN' in df.columns:
        agg_dict['SYSTEM_WIDE_GEN'] = 'mean'
    if 'SYSTEM_WIDE_HSL' in df.columns:
        agg_dict['SYSTEM_WIDE_HSL'] = 'mean'
    
    df_hourly = df.groupby('TimestampHour').agg(agg_dict).reset_index()
    
    # Rename to final names
    rename_map = {
        'SYSTEM_WIDE_GEN': 'Solar_Actual_5min_Hourly',
        'SYSTEM_WIDE_HSL': 'Solar_HSL_5min_Hourly'
    }
    df_hourly = df_hourly.rename(columns=rename_map)
    
    logger.info(f"  ✓ Output: {len(df_hourly)} rows (system-wide)")
    
    return df_hourly


# ----------------------------------------------------------------------------
# 8. hist_SolarPowerProductionHourlyAveragedActualandForecastedValues
# ----------------------------------------------------------------------------

def process_solar_hourly(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Solar hourly actual + forecasts.
    Columns: SYSTEM_WIDE_GEN, COP_HSL_SYSTEM_WIDE, STPPF_SYSTEM_WIDE, 
             PVGRPP_SYSTEM_WIDE, SYSTEM_WIDE_HSL
    """
    logger.info("\n[8/9] Processing hist_SolarPowerProductionHourlyAveragedActualandForecastedValues")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_SolarPowerProductionHourlyAveragedActualandForecastedValues]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_deliverydate_hourending(df)
    
    # Rename to final names
    rename_map = {
        'SYSTEM_WIDE_GEN': 'Solar_Actual_Hourly',
        'COP_HSL_SYSTEM_WIDE': 'Solar_COP_HSL_Hourly',
        'STPPF_SYSTEM_WIDE': 'Solar_Forecast_STPPF_Hourly',
        'PVGRPP_SYSTEM_WIDE': 'Solar_Forecast_PVGRPP_Hourly',
        'SYSTEM_WIDE_HSL': 'Solar_HSL_Hourly'
    }
    
    df = df.rename(columns=rename_map)
    
    # Keep only timestamp + renamed columns
    keep_cols = ['TimestampHour'] + [v for v in rename_map.values() if v in df.columns]
    df = df[keep_cols]
    
    logger.info(f"  ✓ Output: {len(df)} rows (system-wide)")
    
    return df


# ----------------------------------------------------------------------------
# 9. hist_WindPowerProductionHourlyAveragedActualandForecastedValues
# ----------------------------------------------------------------------------

def process_wind_hourly(conn: pyodbc.Connection) -> pd.DataFrame:
    """
    Wind hourly actual + forecasts (system + 3 regions).
    Must melt regional columns into long format.
    
    System columns: SYSTEM_WIDE_GEN, COP_HSL_SYSTEM_WIDE, STWPF_SYSTEM_WIDE, 
                    WGRPP_SYSTEM_WIDE, SYSTEM_WIDE_HSL
    
    Regional columns (LZ_SOUTH_HOUSTON, LZ_WEST, LZ_NORTH):
    - GEN_LZ_{region}
    - COP_HSL_LZ_{region}
    - STWPF_LZ_{region}
    - WGRPP_LZ_{region}
    """
    logger.info("\n[9/9] Processing hist_WindPowerProductionHourlyAveragedActualandForecastedValues")
    
    df = load_table_chunked(conn, "[ERCOT].[hist_WindPowerProductionHourlyAveragedActualandForecastedValues]")
    
    if df.empty:
        return pd.DataFrame()
    
    df = normalize_deliverydate_hourending(df)
    
    # Process system-wide columns
    system_rename = {
        'SYSTEM_WIDE_GEN': 'Wind_Actual_System_Hourly',
        'SYSTEM_WIDE_HSL': 'Wind_HSL_System_Hourly',
        'STWPF_SYSTEM_WIDE': 'Wind_Forecast_STWPF_System_Hourly',
        'WGRPP_SYSTEM_WIDE': 'Wind_Forecast_WGRPP_System_Hourly'
    }
    
    df_system = df[['TimestampHour'] + [k for k in system_rename.keys() if k in df.columns]].copy()
    df_system = df_system.rename(columns=system_rename)
    
    # Process regional columns (melt)
    regions = ['SOUTH_HOUSTON', 'WEST', 'NORTH']
    regional_dfs = []
    
    for region in regions:
        cols_to_melt = {
            f'GEN_LZ_{region}': f'Wind_Actual_{region}_Hourly',
            f'COP_HSL_LZ_{region}': f'Wind_HSL_{region}_Hourly',
            f'STWPF_LZ_{region}': f'Wind_Forecast_STWPF_{region}_Hourly',
            f'WGRPP_LZ_{region}': f'Wind_Forecast_WGRPP_{region}_Hourly'
        }
        
        existing_cols = {k: v for k, v in cols_to_melt.items() if k in df.columns}
        
        if existing_cols:
            df_region = df[['TimestampHour'] + list(existing_cols.keys())].copy()
            df_region = df_region.rename(columns=existing_cols)
            regional_dfs.append(df_region)
    
    # Merge system + regions
    result = df_system
    for df_region in regional_dfs:
        result = result.merge(df_region, on='TimestampHour', how='outer')
    
    logger.info(f"  ✓ Output: {len(result)} rows (system + 3 regions)")
    
    return result


# ============================================================================
# MERGE STRATEGY
# ============================================================================

def merge_all_features(
    df_load_forecast: pd.DataFrame,
    df_load_weather: pd.DataFrame,
    df_dam: pd.DataFrame,
    df_rtm: pd.DataFrame,
    df_rtm_alt: pd.DataFrame,
    df_sced: pd.DataFrame,
    df_solar_5min: pd.DataFrame,
    df_solar_hourly: pd.DataFrame,
    df_wind: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all feature tables into one unified hourly DataFrame.
    
    Strategy:
    - Start with settlement point prices (DAM + RTM) which define the grain
    - Add global features (load, solar, wind) by broadcasting on TimestampHour
    - Keep data in long format: one row per hour per SettlementPoint
    - This preserves the ability to predict DART spread per settlement point
    """
    logger.info("\n" + "="*80)
    logger.info("MERGING ALL FEATURES")
    logger.info("="*80)
    
    # Start with DAM prices (defines the grain: hour x settlement point)
    if df_dam.empty:
        logger.error("DAM prices table is empty - cannot proceed")
        return pd.DataFrame()
    
    logger.info(f"Starting with DAM prices: {df_dam.shape}")
    merged = df_dam.copy()
    
    # Merge RTM LMP (on TimestampHour + SettlementPoint)
    if not df_rtm.empty:
        logger.info(f"Merging RTM LMP: {df_rtm.shape}")
        merged = merged.merge(
            df_rtm,
            on=['TimestampHour', 'SettlementPoint'],
            how='outer'
        )
        logger.info(f"  After RTM merge: {merged.shape}")
    
    # Merge alternative RTM if available
    if not df_rtm_alt.empty:
        logger.info(f"Merging alternative RTM: {df_rtm_alt.shape}")
        merged = merged.merge(
            df_rtm_alt,
            on=['TimestampHour', 'SettlementPoint'],
            how='outer'
        )
        logger.info(f"  After RTM alt merge: {merged.shape}")
    
    # Now add global features (load, solar, wind) by broadcasting on TimestampHour
    logger.info("\nAdding global features (broadcast on TimestampHour)...")
    
    global_dfs = [
        ('Load Forecast', df_load_forecast),
        ('Load Weather', df_load_weather),
        ('Solar 5min', df_solar_5min),
        ('Solar Hourly', df_solar_hourly),
        ('Wind', df_wind)
    ]
    
    for name, df in global_dfs:
        if df.empty:
            logger.info(f"  Skipping {name} (empty)")
            continue
        
        logger.info(f"  Merging {name}: {df.shape}")
        merged = merged.merge(df, on='TimestampHour', how='left')
        logger.info(f"    After merge: {merged.shape}")
    
    # Add constraint features (optional - sparse data)
    # Note: This creates even more rows, so we'll skip for now
    # In production, constraints could be aggregated to system-level metrics
    
    logger.info("\n" + "="*80)
    logger.info(f"✓ Final merged dataset: {merged.shape}")
    logger.info(f"  Date range: {merged['TimestampHour'].min()} to {merged['TimestampHour'].max()}")
    logger.info(f"  Settlement points: {merged['SettlementPoint'].nunique()}")
    logger.info(f"  Total features: {len(merged.columns) - 2}")  # -2 for TimestampHour and SettlementPoint
    logger.info("="*80)
    
    return merged


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main feature engineering pipeline.
    
    Steps:
    1. Connect to SQL Server
    2. Load and process all 9 ERCOT tables
    3. Normalize timestamps
    4. Resample 5-minute data to hourly
    5. Melt wide tables to long format
    6. Rename columns to engineered feature names
    7. Merge into unified hourly DataFrame
    8. Write to Azure ML output folder
    """
    try:
        logger.info("="*80)
        logger.info("ERCOT FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get Azure ML output path
        output_folder = os.environ.get("AZUREML_OUTPUT_features")
        
        if not output_folder:
            logger.warning("AZUREML_OUTPUT_features not found, using local path")
            output_folder = "data/features"
        
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")
        
        # Connect to SQL Server
        logger.info("\n" + "="*80)
        logger.info("SQL SERVER CONNECTION")
        logger.info("="*80)
        conn = get_sql_connection()
        
        # Process all tables
        logger.info("\n" + "="*80)
        logger.info("LOADING AND PROCESSING TABLES")
        logger.info("="*80)
        
        df_load_forecast = process_load_forecast_zone(conn)
        df_load_weather = process_load_weather_zone(conn)
        df_dam = process_dam_prices(conn)
        df_rtm = process_rtm_lmp(conn)
        df_rtm_alt = process_rtm_lmp_alt(conn)
        df_sced = process_sced_constraints(conn)
        df_solar_5min = process_solar_5min(conn)
        df_solar_hourly = process_solar_hourly(conn)
        df_wind = process_wind_hourly(conn)
        
        # Close connection
        conn.close()
        logger.info("\n✓ SQL connection closed")
        
        # Merge all features
        merged = merge_all_features(
            df_load_forecast,
            df_load_weather,
            df_dam,
            df_rtm,
            df_rtm_alt,
            df_sced,
            df_solar_5min,
            df_solar_hourly,
            df_wind
        )
        
        if merged.empty:
            raise ValueError("Merged dataset is empty!")
        
        # Sort by timestamp
        merged = merged.sort_values('TimestampHour').reset_index(drop=True)
        
        # Save to parquet
        logger.info("\n" + "="*80)
        logger.info("SAVING TO PARQUET")
        logger.info("="*80)
        
        output_file = os.path.join(output_folder, "hourly_features.parquet")
        merged.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        
        logger.info(f"✓ Saved to: {output_file}")
        logger.info(f"  Rows: {len(merged):,}")
        logger.info(f"  Columns: {len(merged.columns)}")
        logger.info(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)
        
        logger.info(f"Date range: {merged['TimestampHour'].min()} to {merged['TimestampHour'].max()}")
        logger.info(f"Total hours: {len(merged):,}")
        logger.info(f"Total features: {len(merged.columns) - 1}")
        
        # Missing value analysis
        missing_pct = (merged.isna().sum() / len(merged) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            logger.info(f"\nFeatures with >50% missing values:")
            for col, pct in high_missing.items():
                logger.info(f"  {col}: {pct}%")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("PIPELINE FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
