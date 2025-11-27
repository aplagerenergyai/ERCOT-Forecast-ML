# Local Weighted Ensemble Guide

## Quick Start (Today)

### 1. Download Models from Azure ML

```bash
# Run the ensemble script - it will download all 9 models
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --output ensemble_predictions.csv
```

**First run:** ~5-10 minutes (downloading 9 models)
**Subsequent runs:** ~2-3 minutes (using `--skip-download`)

### 2. Use Existing Models (Skip Download)

```bash
# If you already ran it once
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --output ensemble_predictions.csv \
  --skip-download
```

### 3. Output

The script creates `ensemble_predictions.csv` with:
- `actual`: True DART spread values
- `ensemble_prediction`: Weighted average prediction
- `ensemble_val_rmse`: Ensemble validation RMSE
- `lgbm_prediction`, `xgb_prediction`, etc.: Individual model predictions
- `lgbm_val_rmse`, `xgb_val_rmse`, etc.: Individual model RMSEs

## What the Script Does

### Smart Weighting
Models are weighted by **inverse RMSE** - better models get higher weight:
- If LightGBM has RMSE=$10 and RandomForest has RMSE=$20
- LightGBM gets 2x the weight of RandomForest
- All weights sum to 1.0

### Example Output
```
Model weights (based on validation RMSE):
  lgbm           : 0.152 (RMSE: $89.32)
  xgb            : 0.148 (RMSE: $91.75)
  catboost       : 0.142 (RMSE: $95.42)
  histgb         : 0.135 (RMSE: $100.28)
  tabnet         : 0.128 (RMSE: $105.91)
  deep           : 0.115 (RMSE: $118.23)
  extratrees     : 0.095 (RMSE: $142.87)
  rf             : 0.085 (RMSE: $159.64)
  automl         : 0.082 (RMSE: $165.52)

RESULTS:
  Best single model RMSE: $89.32 (lgbm)
  Ensemble RMSE:          $86.15
  Improvement:            3.55%
```

## Troubleshooting

### "Need at least 2 models"
- Check you're logged into Azure: `az login`
- Verify job IDs in `MODEL_JOBS` dict in script
- Check models exist in Azure ML workspace

### "Out of memory"
```bash
# Reduce sample size
python local_ensemble.py \
  --features data/hourly_features.parquet \
  --max-samples 1000000  # 1M instead of 2M
```

### "az: command not found"
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Or on macOS
brew install azure-cli
```

---

# Feature Engineering Guide (Tomorrow)

## High-Impact Features to Add

### 1. Lagged Features (30 min)
```python
# Previous values
df['DART_lag_1h'] = df.groupby('SettlementPoint')['DART_Spread'].shift(1)
df['DART_lag_24h'] = df.groupby('SettlementPoint')['DART_Spread'].shift(24)
df['DART_lag_168h'] = df.groupby('SettlementPoint')['DART_Spread'].shift(168)  # 1 week
```

### 2. Rolling Statistics (30 min)
```python
# Moving averages
df['DART_ma_7d'] = df.groupby('SettlementPoint')['DART_Spread'].rolling(168).mean().reset_index(0, drop=True)
df['DART_ma_30d'] = df.groupby('SettlementPoint')['DART_Spread'].rolling(720).mean().reset_index(0, drop=True)

# Volatility
df['DART_std_7d'] = df.groupby('SettlementPoint')['DART_Spread'].rolling(168).std().reset_index(0, drop=True)
```

### 3. Time-Based Features (20 min)
```python
# Hour interactions
df['hour_weekend'] = df['Hour'] * df['IsWeekend']
df['hour_season'] = df['Hour'] * df['Quarter']

# Cyclic encoding (important for hours!)
df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
```

### 4. Price Interactions (20 min)
```python
# Spread between RTP and DAM
df['RTM_DAM_spread'] = df['Settlement_Point_Price'] - df['DAM_Price']

# Price ratios
df['price_to_load_ratio'] = df['Settlement_Point_Price'] / (df['System_Load'] + 1)
df['wind_to_load_ratio'] = df['Total_Wind_Output'] / (df['System_Load'] + 1)
```

### 5. Weather-Energy Interactions (20 min)
```python
# Temperature impact on load
df['temp_load_interaction'] = df['Temperature'] * df['System_Load']

# Wind conditions
df['wind_capacity_factor'] = df['Total_Wind_Output'] / (df['Wind_HSL_Hourly'] + 1)

# Solar utilization
df['solar_capacity_factor'] = df['Total_Solar_Output'] / (df['Solar_HSL_Hourly'] + 1)
```

## Implementation Plan (Tomorrow)

### Morning (2-3 hours)
1. Add lagged features to `build_features.py`
2. Add rolling statistics
3. Test locally: `python build_features.py`

### Afternoon (2-3 hours)
4. Add time-based and interaction features
5. Rebuild features in Azure: `az ml job create --file aml_build_features.yml`
6. Retrain top 3 models (LightGBM, XGBoost, CatBoost)

### Evening (1-2 hours)
7. Compare old vs new model performance
8. Run local ensemble with new models
9. Document improvements

## Expected Results

**Conservative estimate:** 10-15% RMSE improvement
**Optimistic estimate:** 20-30% RMSE improvement

Lagged features alone typically improve time series models by 10-20%.

---

## Questions?

**Today:** Focus on getting the local ensemble working
**Tomorrow:** Feature engineering to improve all models

The ensemble gives you the quick win now. Feature engineering gives you the big win tomorrow! 🚀

