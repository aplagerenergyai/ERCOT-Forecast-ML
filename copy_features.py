"""
Simple script to copy features from one Azure ML location to another.
This ensures the file is in a location that can be mounted by training jobs.
"""
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("COPYING FEATURES TO MOUNTABLE LOCATION")
    logger.info("="*80)
    
    # Input: The features from the previous job (mounted)
    input_path = os.environ.get("AZURE_ML_INPUT_FEATURES", os.environ.get("AZUREML_DATAREFERENCE_FEATURES"))
    if not input_path:
        logger.error("❌ No input path found")
        return
    
    logger.info(f"Input path: {input_path}")
    
    # Output: Standard Azure ML output that will be properly registered
    output_path = "./outputs"
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Output path: {output_path}")
    
    # Find the parquet file in input
    input_file = os.path.join(input_path, "hourly_features.parquet")
    output_file = os.path.join(output_path, "hourly_features.parquet")
    
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        logger.info(f"Contents of input directory: {os.listdir(input_path)}")
        return
    
    logger.info(f"✓ Found input file: {input_file}")
    
    # Copy the file
    logger.info("Copying file...")
    shutil.copy2(input_file, output_file)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"✅ SUCCESS! File copied to: {output_file}")
    logger.info(f"   Size: {file_size_mb:.2f} MB")
    logger.info("="*80)

if __name__ == "__main__":
    main()

