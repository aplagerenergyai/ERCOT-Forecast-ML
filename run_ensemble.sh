#!/bin/bash
# Quick script to run the local ensemble on your desktop

set -e  # Exit on error

echo "========================================"
echo "ERCOT ML - Local Ensemble Runner"
echo "========================================"
echo ""

# Check prerequisites
echo "✓ Checking prerequisites..."

if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Please install: https://aka.ms/installazurecliwindows"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "⚠️  Not logged into Azure. Running 'az login'..."
    az login
fi

echo "✓ Prerequisites OK"
echo ""

# Check for features file
if [ ! -f "data/hourly_features.parquet" ]; then
    echo "⚠️  Features file not found at: data/hourly_features.parquet"
    echo ""
    echo "Please download it manually:"
    echo "1. Go to Azure ML Studio: https://ml.azure.com"
    echo "2. Navigate to Data > ercot_features_manual"
    echo "3. Click Download"
    echo "4. Save to: data/hourly_features.parquet"
    echo ""
    read -p "Press Enter when ready, or Ctrl+C to cancel..."
fi

if [ ! -f "data/hourly_features.parquet" ]; then
    echo "❌ Still no features file. Exiting."
    exit 1
fi

echo "✓ Features file found"
echo ""

# Ask user what to do
if [ -d "downloaded_models" ] && [ "$(ls -A downloaded_models)" ]; then
    echo "📁 Found existing models in downloaded_models/"
    echo ""
    echo "Options:"
    echo "  1) Use existing models (FAST - 2-3 minutes)"
    echo "  2) Re-download models (SLOW - 10-15 minutes)"
    echo ""
    read -p "Choose [1/2] (default: 1): " choice
    choice=${choice:-1}
    
    if [ "$choice" = "1" ]; then
        SKIP_FLAG="--skip-download"
        echo "✓ Using existing models"
    else
        SKIP_FLAG=""
        echo "✓ Will re-download models"
    fi
else
    SKIP_FLAG=""
    echo "📥 No existing models found - will download"
fi

echo ""
echo "========================================"
echo "Running Ensemble..."
echo "========================================"
echo ""

# Run the ensemble
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --output ensemble_predictions.csv \
  $SKIP_FLAG

echo ""
echo "========================================"
echo "✅ ENSEMBLE COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: ensemble_predictions.csv"
echo ""
echo "To view results:"
echo "  - Open ensemble_predictions.csv in Excel"
echo "  - Or run: head -20 ensemble_predictions.csv"
echo ""
echo "Next step: Feature engineering (see LOCAL_ENSEMBLE_GUIDE.md)"
echo ""

