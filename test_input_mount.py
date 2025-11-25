import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_aml_input_mount(input_name: str = "features"):
    logger.info("="*80)
    logger.info(f"STARTING AZURE ML INPUT MOUNT TEST for input '{input_name}'")
    logger.info("="*80)

    # --- Step 1: Check for environment variables ---
    logger.info("\n🔍 Checking environment variables for input path...")
    possible_env_vars = [
        f"AZURE_ML_INPUT_{input_name.upper()}",
        f"AZUREML_DATAREFERENCE_{input_name}",
        f"AZUREML_INPUT_{input_name}"
    ]
    
    input_path = None
    for env_var in possible_env_vars:
        path = os.environ.get(env_var)
        if path:
            logger.info(f"  ✓ Found environment variable '{env_var}': {path}")
            input_path = path
            break
        else:
            logger.info(f"  ✗ Environment variable '{env_var}' not found.")

    if not input_path:
        logger.error("❌ No Azure ML input environment variable found for 'features'.")
        logger.info("="*80)
        return False

    # --- Step 2: Verify the mounted directory exists ---
    logger.info(f"\n📂 Verifying mounted directory exists: {input_path}")
    if not os.path.isdir(input_path):
        logger.error(f"❌ Mounted directory does not exist: {input_path}")
        logger.info("  Contents of parent directory:")
        try:
            logger.info(os.listdir(os.path.dirname(input_path)))
        except Exception as e:
            logger.error(f"  Could not list parent directory: {e}")
        logger.info("="*80)
        return False
    logger.info(f"  ✓ Mounted directory exists: {input_path}")

    # --- Step 3: List contents of the mounted directory ---
    logger.info(f"\n📄 Listing contents of mounted directory: {input_path}")
    dir_contents = os.listdir(input_path)
    if not dir_contents:
        logger.error(f"❌ Mounted directory is empty: {input_path}")
        logger.info("  This means the feature engineering job did not save the file correctly to this output.")
        logger.info("="*80)
        return False
    logger.info(f"  ✓ Contents: {dir_contents}")

    # --- Step 4: Check for the specific parquet file ---
    parquet_file_name = "hourly_features.parquet"
    parquet_full_path = os.path.join(input_path, parquet_file_name)
    logger.info(f"\n🔎 Checking for parquet file: {parquet_full_path}")
    if not os.path.exists(parquet_full_path):
        logger.error(f"❌ Parquet file not found in mounted directory: {parquet_full_path}")
        logger.info("  Ensure the feature engineering script saves the file with this name.")
        logger.info("="*80)
        return False
    logger.info(f"  ✓ Parquet file found: {parquet_full_path}")

    # --- Step 5: Attempt to load the parquet file ---
    logger.info(f"\n📊 Attempting to load parquet file: {parquet_full_path}")
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_full_path)
        logger.info(f"  ✓ Successfully loaded parquet file. Rows: {len(df):,}, Columns: {len(df.columns)}")
        logger.info(f"  Date Range: {df['TimestampHour'].min()} to {df['TimestampHour'].max()}")
    except Exception as e:
        logger.error(f"❌ Error loading parquet file: {e}")
        logger.info("  This could indicate a corrupted file or missing dependencies.")
        logger.info("="*80)
        return False

    logger.info("\n" + "="*80)
    logger.info("✅ ✅ ✅  ALL AZURE ML INPUT MOUNT TESTS PASSED  ✅ ✅ ✅")
    logger.info("="*80)
    return True

if __name__ == "__main__":
    test_aml_input_mount()
