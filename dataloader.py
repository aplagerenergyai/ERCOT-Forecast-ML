"""
dataloader.py

Loads the unified hourly features parquet from Azure ML storage,
performs time-based train/val/test split, encodes categorical features,
and standardizes continuous features for ML model training.
"""

import os
import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERCOTDataLoader:
    """
    Data loader for ERCOT hourly features.
    
    Handles:
    - Loading parquet from Azure ML storage
    - Creating DART target (DAM - RTM spread)
    - Time-based train/val/test split
    - Categorical encoding
    - Feature standardization
    """
    
    def __init__(self, features_path: str):
        """
        Args:
            features_path: Path to hourly_features.parquet
        """
        self.features_path = features_path
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.continuous_columns = []
        
    def load_data(self, sample_rows: int = None) -> pd.DataFrame:
        """
        Load the parquet file from Azure ML storage.
        
        Args:
            sample_rows: Optional - sample this many rows during load (memory efficient)
        """
        logger.info(f"Loading features from: {self.features_path}")
        
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        df = pd.read_parquet(self.features_path)
        logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Sample immediately after load if requested (before any processing)
        if sample_rows is not None and len(df) > sample_rows:
            logger.info(f"⚠️  Sampling dataset during load: {len(df):,} → {sample_rows:,} rows")
            sample_indices = np.random.RandomState(42).choice(
                len(df), size=sample_rows, replace=False
            )
            sample_indices.sort()  # Maintain temporal order
            df = df.iloc[sample_indices].copy()
            logger.info(f"✓ Dataset sampled to: {len(df):,} rows")
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DART target: DAM_Price_Hourly - RTM_LMP_HourlyAvg
        Drop rows where either component is missing.
        """
        logger.info("Creating DART target variable...")
        
        # Check for required columns
        dam_col = 'DAM_Price_Hourly'
        rtm_col = 'RTM_LMP_HourlyAvg'
        
        if dam_col not in df.columns:
            raise ValueError(f"Missing required column: {dam_col}")
        if rtm_col not in df.columns:
            raise ValueError(f"Missing required column: {rtm_col}")
        
        # Create target
        df['DART'] = df[dam_col] - df[rtm_col]
        
        # Check for missing values
        before_count = len(df)
        df = df.dropna(subset=[dam_col, rtm_col, 'DART'])
        after_count = len(df)
        
        logger.info(f"  Dropped {before_count - after_count:,} rows with missing price data")
        logger.info(f"  Final dataset: {after_count:,} rows")
        logger.info(f"  DART statistics:")
        logger.info(f"    Mean: ${df['DART'].mean():.2f}")
        logger.info(f"    Std: ${df['DART'].std():.2f}")
        logger.info(f"    Min: ${df['DART'].min():.2f}")
        logger.info(f"    Max: ${df['DART'].max():.2f}")
        
        return df
    
    def time_based_split(
        self, 
        df: pd.DataFrame, 
        train_pct: float = 0.8, 
        val_pct: float = 0.1, 
        test_pct: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform time-based split (no shuffling).
        
        Args:
            df: DataFrame with TimestampHour column
            train_pct: Training set percentage (default 0.8)
            val_pct: Validation set percentage (default 0.1)
            test_pct: Test set percentage (default 0.1)
        
        Returns:
            train_df, val_df, test_df
        """
        logger.info(f"Performing time-based split: {train_pct:.0%}/{val_pct:.0%}/{test_pct:.0%}")
        
        # Sort by timestamp
        df = df.sort_values('TimestampHour').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"  Train: {len(train_df):,} rows ({train_df['TimestampHour'].min()} to {train_df['TimestampHour'].max()})")
        logger.info(f"  Val:   {len(val_df):,} rows ({val_df['TimestampHour'].min()} to {val_df['TimestampHour'].max()})")
        logger.info(f"  Test:  {len(test_df):,} rows ({test_df['TimestampHour'].min()} to {test_df['TimestampHour'].max()})")
        
        return train_df, val_df, test_df
    
    def identify_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify categorical and continuous feature columns.
        
        Exclude:
        - TimestampHour (temporal index)
        - DART (target)
        - DAM_Price_Hourly, RTM_LMP_HourlyAvg (used to create target)
        
        Categorical columns (require encoding):
        - SettlementPoint
        - ConstraintName
        - Any column with 'Region' or 'Zone' in name and low cardinality
        """
        exclude_cols = ['TimestampHour', 'DART', 'DAM_Price_Hourly', 'RTM_LMP_HourlyAvg']
        
        all_feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        categorical_cols = []
        continuous_cols = []
        
        for col in all_feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
            elif df[col].nunique() < 50 and col.lower() in ['settlementpoint', 'constraintname', 'regionid', 'zone']:
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
        
        logger.info(f"Identified {len(all_feature_cols)} total features:")
        logger.info(f"  Categorical: {len(categorical_cols)}")
        logger.info(f"  Continuous: {len(continuous_cols)}")
        
        if categorical_cols:
            logger.info(f"  Categorical columns: {categorical_cols}")
        
        return {
            'all': all_feature_cols,
            'categorical': categorical_cols,
            'continuous': continuous_cols
        }
    
    def encode_categorical_features(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features using Target Encoding.
        Fit on training set only, transform all sets.
        """
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return train_df, val_df, test_df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns using TargetEncoder...")
        
        for col in categorical_cols:
            if col not in train_df.columns:
                logger.warning(f"  Skipping {col} - not found in dataframe")
                continue
            
            logger.info(f"  Encoding: {col} ({train_df[col].nunique()} unique values)")
            
            encoder = TargetEncoder(cols=[col], smoothing=1.0)
            
            # Fit on training data only
            encoder.fit(train_df[[col]], train_df['DART'])
            
            # Transform all sets
            train_df[col] = encoder.transform(train_df[[col]])[col]
            val_df[col] = encoder.transform(val_df[[col]])[col]
            test_df[col] = encoder.transform(test_df[[col]])[col]
            
            self.categorical_encoders[col] = encoder
        
        logger.info("✓ Categorical encoding complete")
        
        return train_df, val_df, test_df
    
    def standardize_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        continuous_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standardize continuous features using training set mean/std.
        """
        if not continuous_cols:
            logger.info("No continuous columns to standardize")
            return train_df, val_df, test_df
        
        logger.info(f"Standardizing {len(continuous_cols)} continuous features...")
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[continuous_cols].fillna(0))
        
        # Transform all sets
        train_df[continuous_cols] = self.scaler.transform(train_df[continuous_cols].fillna(0))
        val_df[continuous_cols] = self.scaler.transform(val_df[continuous_cols].fillna(0))
        test_df[continuous_cols] = self.scaler.transform(test_df[continuous_cols].fillna(0))
        
        logger.info("✓ Feature standardization complete")
        
        return train_df, val_df, test_df
    
    def prepare_datasets(
        self,
        max_train_samples: int = None,
        max_total_samples: int = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Full pipeline: [sample during load] → load → target → split → encode → standardize.
        
        Args:
            max_train_samples: Optional limit on training samples after split.
            max_total_samples: Optional limit on total samples - samples during LOAD operation.
                             Most memory efficient - use this for GPU models with severe constraints.
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        logger.info("="*80)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("="*80)
        
        # Load data with optional sampling (most memory efficient)
        df = self.load_data(sample_rows=max_total_samples)
        
        # Create target
        df = self.create_target(df)
        
        # Time-based split
        train_df, val_df, test_df = self.time_based_split(df)
        
        # Memory optimization: Sample training data if requested
        if max_train_samples is not None and len(train_df) > max_train_samples:
            logger.info(f"⚠️  Memory optimization: Training set has {len(train_df):,} rows")
            logger.info(f"   Sampling to {max_train_samples:,} rows")
            sample_indices = np.random.RandomState(42).choice(
                len(train_df), size=max_train_samples, replace=False
            )
            sample_indices.sort()  # Maintain temporal order
            train_df = train_df.iloc[sample_indices].copy()
            logger.info(f"✓ Sampled training set: {len(train_df):,} rows")
        
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
        train_df, val_df, test_df = self.standardize_features(
            train_df, val_df, test_df, self.continuous_columns
        )
        
        # Extract X and y
        X_train = train_df[self.feature_columns].values
        y_train = train_df['DART'].values
        
        X_val = val_df[self.feature_columns].values
        y_val = val_df['DART'].values
        
        X_test = test_df[self.feature_columns].values
        y_test = test_df['DART'].values
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
        logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Features: {len(self.feature_columns)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_features_from_aml_input(input_name: str = "features") -> str:
    """
    Helper function to get the path to the features parquet from Azure ML input.
    
    Args:
        input_name: Name of the input in the Azure ML job (default: "features")
    
    Returns:
        Path to hourly_features.parquet
    """
    # DEBUG: Print all environment variables that contain "AZUREML" or "INPUT"
    logger.info("🔍 Debugging Azure ML input detection:")
    logger.info(f"  Current working directory: {os.getcwd()}")
    logger.info(f"  Looking for input: {input_name}")
    
    relevant_env_vars = {k: v for k, v in os.environ.items() if 'AZUREML' in k or 'INPUT' in k or 'input' in k}
    if relevant_env_vars:
        logger.info("  Relevant environment variables:")
        for key, value in relevant_env_vars.items():
            logger.info(f"    {key} = {value[:100]}...")  # Truncate long paths
    else:
        logger.info("  No AZUREML/INPUT environment variables found")
    
    # List contents of current directory
    try:
        cwd_contents = os.listdir(os.getcwd())
        logger.info(f"  Contents of current directory: {cwd_contents[:10]}")  # First 10 items
    except Exception as e:
        logger.info(f"  Could not list current directory: {e}")
    
    # Try multiple Azure ML v2 input patterns
    possible_paths = [
        # Azure ML v2 standard pattern: AZURE_ML_INPUT_<name>
        os.environ.get(f"AZURE_ML_INPUT_{input_name}"),
        os.environ.get(f"AZURE_ML_INPUT_{input_name.upper()}"),
        os.environ.get(f"AZURE_ML_INPUT_{input_name.lower()}"),
        # Alternative patterns
        os.environ.get(f"azure_ml_input_{input_name}"),
        os.environ.get(f"azure_ml_input_{input_name.lower()}"),
        os.environ.get(f"AZUREML_DATAREFERENCE_{input_name}"),
        os.environ.get(f"AZUREML_DATAREFERENCE_{input_name.upper()}"),
        # Check if mounted at current working directory
        os.path.join(os.getcwd(), input_name),
        os.path.join(os.getcwd(), input_name.lower()),
        os.path.join(os.getcwd(), input_name.upper()),
    ]
    
    logger.info(f"  Checking {len(possible_paths)} possible paths...")
    
    for idx, base_path in enumerate(possible_paths):
        if not base_path:
            continue
        logger.info(f"  [{idx+1}] Checking: {base_path}")
        if os.path.exists(base_path):
            logger.info(f"      ✓ Path exists!")
            # Try to find hourly_features.parquet directly
            parquet_path = os.path.join(base_path, "hourly_features.parquet")
            if os.path.exists(parquet_path):
                logger.info(f"✅ Using Azure ML input path: {parquet_path}")
                return parquet_path
            # Check if it's directly a parquet file
            if os.path.isfile(base_path) and base_path.endswith('.parquet'):
                logger.info(f"✅ Using Azure ML input file: {base_path}")
                return base_path
            # Search recursively for hourly_features.parquet in subdirectories
            logger.info(f"      Searching recursively for hourly_features.parquet...")
            import glob
            recursive_search = glob.glob(os.path.join(base_path, "**/hourly_features.parquet"), recursive=True)
            if recursive_search:
                found_path = recursive_search[0]
                logger.info(f"✅ Found parquet file recursively: {found_path}")
                return found_path
            # List what's in this directory
            try:
                contents = os.listdir(base_path)
                logger.info(f"      Contents: {contents}")
            except:
                pass
        else:
            logger.info(f"      ✗ Path does not exist")
    
    # Fallback for local testing
    local_path = "data/features/hourly_features.parquet"
    logger.warning(f"⚠️  Azure ML input not found after checking all paths, using local path: {local_path}")
    return local_path

