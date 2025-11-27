# Running Ensemble on Your Desktop

## Prerequisites (One-Time Setup)

### 1. Install Python 3.10+
```bash
# Check if you have Python
python --version  # Should be 3.10 or higher
```

### 2. Install Azure CLI
**Windows:**
```powershell
# Download and run installer
https://aka.ms/installazurecliwindows
```

**Linux/WSL:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**macOS:**
```bash
brew install azure-cli
```

### 3. Login to Azure
```bash
az login
```
This will open a browser window - login with: `aplager@softsmiths.com`

### 4. Clone the Repository
```bash
git clone https://github.com/aplagerenergyai/ERCOT-Forecast-ML.git
cd ERCOT-Forecast-ML/forecasting-ml
```

### 5. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Running the Ensemble (Every Time)

### Step 1: Download Features Data (First Time Only)
The features file is ~500MB, so this takes a few minutes:

```bash
# Create data directory
mkdir -p data

# Option A: Download from Azure Storage (if you have access)
az storage blob download \
  --account-name ercotforecastingprod \
  --container-name azureml-blobstore-<container-id> \
  --name LocalUpload/<hash>/hourly_features.parquet \
  --file data/hourly_features.parquet

# Option B: Use a smaller sample for testing (recommended first time)
# The ensemble script will automatically sample to 2M rows anyway
```

**Note:** If you don't have direct blob access, you can also:
1. Download from the Azure ML Studio UI
2. Navigate to Data Assets → ercot_features_manual
3. Click "Download" → save to `data/hourly_features.parquet`

### Step 2: Run the Ensemble Script
```bash
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --output ensemble_predictions.csv
```

**This will:**
1. Download 9 trained models from Azure ML (~5-7 minutes)
   - Models saved to `./downloaded_models/`
   - Total size: ~200-300MB
2. Load and prepare data (~2 minutes)
3. Generate predictions from each model (~3-5 minutes)
4. Create weighted ensemble (~30 seconds)
5. Save results to `ensemble_predictions.csv`

**Total time: 10-15 minutes**

### Step 3: View Results
```bash
# Open the CSV
cat ensemble_predictions.csv | head -20

# Or open in Excel/Sheets
```

## Subsequent Runs (Using Downloaded Models)

After the first run, models are cached locally:

```bash
# Much faster - skips download step
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --output ensemble_predictions.csv \
  --skip-download

# Takes only 2-3 minutes
```

## Understanding the Output

`ensemble_predictions.csv` contains:
- **actual**: True DART spread values (test set)
- **ensemble_prediction**: Weighted average from all 9 models
- **ensemble_val_rmse**: Ensemble validation RMSE
- **lgbm_prediction**: LightGBM individual prediction
- **lgbm_val_rmse**: LightGBM validation RMSE
- **xgb_prediction**: XGBoost individual prediction
- ... (same for all 9 models)

## Folder Structure After Running

```
ERCOT-Forecast-ML/forecasting-ml/
├── data/
│   └── hourly_features.parquet         (~500MB - features dataset)
├── downloaded_models/                   (~300MB total)
│   ├── lgbm/
│   │   └── lgbm_model.pkl
│   ├── xgb/
│   │   └── xgb_model.pkl
│   ├── catboost/
│   │   └── catboost_model.pkl
│   ├── rf/
│   │   └── random_forest_model.pkl
│   ├── deep/
│   │   └── deep_model.pt
│   ├── histgb/
│   │   └── histgb_model.pkl
│   ├── extratrees/
│   │   └── extratrees_model.pkl
│   ├── tabnet/
│   │   └── tabnet_model.pkl
│   └── automl/
│       └── automl_model.pkl
├── ensemble_predictions.csv            (~5MB - results)
└── local_ensemble.py                   (the script)
```

**Total disk space needed: ~1GB**

## Troubleshooting

### "az: command not found"
Install Azure CLI (see Prerequisites above)

### "Authentication failed"
```bash
az login
az account set --subscription "b7788659-1f79-4e40-b98a-eea87041561f"
```

### "Out of memory"
Use fewer samples:
```bash
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --max-samples 1000000  # Use 1M instead of 2M
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Download is stuck
The first download takes 5-10 minutes. If truly stuck:
- Cancel (Ctrl+C)
- Run again with `--skip-download` to use partially downloaded models
- Or delete `downloaded_models/` and start fresh

## Quick Test (2 minutes)

Want to verify everything works quickly?

```bash
# Test with very small sample
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --max-samples 100000 \
  --output test_predictions.csv

# Should complete in ~2 minutes
```

---

## Need Help?

1. Check `LOCAL_ENSEMBLE_GUIDE.md` for more details
2. Check `PROJECT_STATUS.md` for what was accomplished
3. Logs are printed to console - save them if errors occur

