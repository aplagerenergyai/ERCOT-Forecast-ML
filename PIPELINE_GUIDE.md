# ERCOT DART Spread Prediction - ML Pipeline Guide

This guide explains how to run the complete Azure ML pipeline for ERCOT DART spread prediction.

## Architecture Overview

The pipeline consists of two main steps:

### Step 1: Feature Engineering (`aml_build_features.yml`)
- Connects to SQL Server and loads 9 ERCOT historical tables
- Normalizes timestamps to hourly granularity
- Resamples 5-minute data to hourly averages
- Melts wide tables into long format
- Merges all features into a unified parquet file
- **Output**: `hourly_features.parquet` in workspaceblobstore

### Step 2: Model Training (3 parallel jobs)
- **LightGBM** (`aml_train_lgbm.yml`): Gradient boosting tree model
- **XGBoost** (`aml_train_xgb.yml`): Extreme gradient boosting
- **Deep Learning** (`aml_train_deep.yml`): LSTM neural network

All models predict **DART spread** = `DAM_Price_Hourly - RTM_LMP_HourlyAvg`

---

## Data Flow

```
SQL Server (ERCOT Tables)
    ↓
[build_features.py]
    ↓
hourly_features.parquet (in workspaceblobstore)
    ↓
[train_lgbm.py, train_xgb.py, train_deep.py]
    ↓
Trained Models (lgbm_model.pkl, xgb_model.pkl, deep_model.pt)
```

---

## Prerequisites

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Or create conda environment:
```bash
conda env create -f environment.yml
conda activate forecast-env
```

### 2. SQL Server Configuration

Create a `.env` file in the project root:
```env
SQL_SERVER=your-server.database.windows.net
SQL_DATABASE=ERCOT
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
```

### 3. Azure ML Setup

Ensure you have:
- Azure ML workspace created
- Compute clusters configured:
  - `cpu-cluster` (for feature engineering, LightGBM, XGBoost)
  - `gpu-cluster` (for deep learning)
- Default datastore `workspaceblobstore` accessible

---

## Running the Pipeline

### Step 1: Build Features

Submit the feature engineering job:

```bash
az ml job create --file aml_build_features.yml --web
```

This will:
- Load all 9 ERCOT tables from SQL Server
- Process ~100M+ rows of 5-minute data
- Create hourly features with proper timestamp normalization
- Save to: `azureml://datastores/workspaceblobstore/paths/features/`

**Expected runtime**: 30-60 minutes (depending on data size)

**Output**:
- `hourly_features.parquet` (~1-5 GB)
- One row per hour per settlement point (~1053 settlement points)

### Step 2: Train Models (Individual)

Train each model separately:

```bash
# LightGBM
az ml job create --file aml_train_lgbm.yml --web

# XGBoost
az ml job create --file aml_train_xgb.yml --web

# Deep Learning (LSTM)
az ml job create --file aml_train_deep.yml --web
```

**Expected runtime**:
- LightGBM: 10-20 minutes
- XGBoost: 10-20 minutes
- Deep Learning: 30-60 minutes (with GPU)

### Step 3: Train All Models in Parallel

Use the pipeline YAML to train all three models at once:

```bash
az ml job create --file aml_training_pipeline.yml --web
```

This runs all three training jobs in parallel, saving time.

---

## Model Details

### Target Variable: DART Spread

```python
DART = DAM_Price_Hourly - RTM_LMP_HourlyAvg
```

The Day-Ahead vs Real-Time price spread is a key indicator of market efficiency and forecasting accuracy.

### Features

All engineered features from the 9 ERCOT tables:

**Load Features** (13 zones):
- `Load_NORTH_Hourly`, `Load_SOUTH_Hourly`, `Load_WEST_Hourly`, etc.

**Solar Features** (system-wide):
- `Solar_Actual_Hourly`
- `Solar_HSL_Hourly` (High Sustained Limit - uncurtailed potential)
- `Solar_Forecast_STPPF_Hourly`, `Solar_Forecast_PVGRPP_Hourly`
- `Solar_COP_HSL_Hourly`

**Wind Features** (system + 3 regions):
- `Wind_Actual_System_Hourly`, `Wind_HSL_System_Hourly`
- `Wind_Actual_SOUTH_HOUSTON_Hourly`, `Wind_Actual_WEST_Hourly`, etc.
- Forecasts: `Wind_Forecast_STWPF_*`, `Wind_Forecast_WGRPP_*`

**Categorical Features** (encoded):
- `SettlementPoint` (1053 unique values)

### Data Split

Time-based split (no shuffling):
- **Training**: 80% (earliest data)
- **Validation**: 10% (middle period)
- **Test**: 10% (most recent data)

### Feature Engineering

1. **Categorical Encoding**: Target encoding for `SettlementPoint`
2. **Standardization**: Z-score normalization for all continuous features
3. **Missing Value Handling**: Fill with 0 after standardization

---

## Evaluation Metrics

All models are evaluated using:
- **RMSE**: Root Mean Squared Error ($/MWh)
- **MAE**: Mean Absolute Error ($/MWh)
- **MAPE**: Mean Absolute Percentage Error (%)
- **R²**: Coefficient of determination

Metrics are computed for:
- Training set
- Validation set
- Test set

---

## Model Outputs

Each training job saves to workspaceblobstore:

### LightGBM
Path: `azureml://datastores/workspaceblobstore/paths/models/lgbm/`
- `lgbm_model.pkl`: Pickled model + metadata

### XGBoost
Path: `azureml://datastores/workspaceblobstore/paths/models/xgb/`
- `xgb_model.pkl`: Pickled model + metadata

### Deep Learning
Path: `azureml://datastores/workspaceblobstore/paths/models/deep/`
- `deep_model.pt`: PyTorch state dict + metadata

Each model file includes:
- Trained model
- Feature column names
- Scaler (StandardScaler)
- Categorical encoders
- Train/val/test metrics

---

## Troubleshooting

### Issue: "Missing SQL Server credentials"
**Solution**: Ensure `.env` file exists with all 4 variables:
```env
SQL_SERVER=...
SQL_DATABASE=...
SQL_USERNAME=...
SQL_PASSWORD=...
```

### Issue: "AZUREML_OUTPUT_features not found"
**Solution**: The script will fall back to local path `data/features/`. Ensure you're running in Azure ML, not locally.

### Issue: "No data found in table"
**Solution**: Check SQL connection and table names. Ensure ODBC Driver 18 is installed.

### Issue: "DART target has all NaN values"
**Solution**: Verify that both `DAM_Price_Hourly` and `RTM_LMP_HourlyAvg` columns exist and have overlapping data.

### Issue: "GPU cluster not found"
**Solution**: Either create a GPU cluster or change `aml_train_deep.yml` to use `cpu-cluster` (training will be slower).

---

## Customization

### Changing Hyperparameters

Edit the training scripts directly:

**LightGBM** (`train_lgbm.py`):
```python
params = {
    'num_leaves': 31,        # Increase for more complexity
    'learning_rate': 0.05,   # Decrease for better convergence
    # ...
}
```

**XGBoost** (`train_xgb.py`):
```python
params = {
    'max_depth': 6,          # Increase for deeper trees
    'learning_rate': 0.05,   # Adjust learning rate
    # ...
}
```

**Deep Learning** (`train_deep.py`):
```python
model = train_deep_model(
    epochs=50,               # Increase for more training
    batch_size=256,          # Adjust based on GPU memory
    learning_rate=0.001      # Tune learning rate
)
```

### Adding More Features

1. Update `build_features.py` to include additional tables
2. Add processor functions following the existing pattern
3. Update `merge_all_features()` to include the new tables

### Changing Train/Val/Test Split

Edit `dataloader.py`:
```python
train_df, val_df, test_df = self.time_based_split(
    df, 
    train_pct=0.8,   # Change to 0.7 for 70/15/15 split
    val_pct=0.1,
    test_pct=0.1
)
```

---

## File Structure

```
forecasting-ml/
├── build_features.py              # Feature engineering script
├── dataloader.py                  # Data loading and preprocessing
├── metrics.py                     # Evaluation metrics
├── train_lgbm.py                  # LightGBM training
├── train_xgb.py                   # XGBoost training
├── train_deep.py                  # Deep learning training
├── aml_build_features.yml         # Azure ML job: feature engineering
├── aml_train_lgbm.yml             # Azure ML job: LightGBM
├── aml_train_xgb.yml              # Azure ML job: XGBoost
├── aml_train_deep.yml             # Azure ML job: Deep learning
├── aml_training_pipeline.yml      # Azure ML pipeline: all models
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── .env                           # SQL credentials (create this)
└── PIPELINE_GUIDE.md             # This file
```

---

## Next Steps

After training models:

1. **Compare Model Performance**: Review test metrics to select the best model
2. **Deploy**: Register the best model in Azure ML and deploy to an endpoint
3. **Monitor**: Set up data drift detection and model performance monitoring
4. **Retrain**: Schedule periodic retraining as new ERCOT data becomes available

---

## Support

For issues or questions:
- Check Azure ML job logs in the Azure Portal
- Review Python script logs for detailed error messages
- Verify SQL Server connectivity and data availability

