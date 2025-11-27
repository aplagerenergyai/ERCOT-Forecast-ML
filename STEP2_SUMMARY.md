# Step 2: ML Training Pipeline - Implementation Summary

## ✅ Files Created

### Python Training Scripts (5 files)
1. **`dataloader.py`** (250 lines)
   - `ERCOTDataLoader` class for data loading and preprocessing
   - Time-based train/val/test split (80/10/10)
   - Target creation: DART = DAM_Price_Hourly - RTM_LMP_HourlyAvg
   - Categorical encoding using TargetEncoder
   - Feature standardization using StandardScaler
   - Automatic feature identification (categorical vs continuous)

2. **`metrics.py`** (90 lines)
   - `calculate_rmse()`: Root Mean Squared Error
   - `calculate_mae()`: Mean Absolute Error
   - `calculate_mape()`: Mean Absolute Percentage Error
   - `calculate_r2()`: R-squared score
   - `evaluate_model()`: Comprehensive evaluation
   - MLflow integration for metric logging

3. **`train_lgbm.py`** (130 lines)
   - LightGBM gradient boosting regressor
   - Early stopping with 50-round patience
   - Feature importance analysis (top 20)
   - Model serialization with metadata
   - Saves to: `outputs/lgbm_model.pkl`

4. **`train_xgb.py`** (130 lines)
   - XGBoost extreme gradient boosting
   - Early stopping with 50-round patience
   - Feature importance by gain
   - Model serialization with metadata
   - Saves to: `outputs/xgb_model.pkl`

5. **`train_deep.py`** (180 lines)
   - PyTorch LSTM regression model
   - Architecture: LSTM (2 layers, 128 hidden dim) → FC layers
   - GPU support with automatic device detection
   - Learning rate scheduling (ReduceLROnPlateau)
   - Early stopping with 10-epoch patience
   - Saves to: `outputs/deep_model.pt`

### Azure ML Job Definitions (4 YAML files)

6. **`aml_train_lgbm.yml`**
   - Compute: cpu-cluster
   - Input: features from workspaceblobstore
   - Output: model to workspaceblobstore/models/lgbm/

7. **`aml_train_xgb.yml`**
   - Compute: cpu-cluster
   - Input: features from workspaceblobstore
   - Output: model to workspaceblobstore/models/xgb/

8. **`aml_train_deep.yml`**
   - Compute: gpu-cluster
   - Input: features from workspaceblobstore
   - Output: model to workspaceblobstore/models/deep/

9. **`aml_training_pipeline.yml`**
   - Parallel execution of all 3 training jobs
   - Shared input: features folder
   - Separate outputs for each model

### Supporting Files

10. **`requirements.txt`**
    - Complete Python dependencies
    - pandas, numpy, pyarrow
    - pyodbc, sqlalchemy
    - lightgbm, xgboost
    - torch, category-encoders
    - azureml-core

11. **`environment.yml`** (updated)
    - Added category-encoders dependency

12. **`PIPELINE_GUIDE.md`**
    - Complete user documentation
    - Step-by-step instructions
    - Troubleshooting guide
    - Architecture overview

---

## 🔄 Updated Files

### `build_features.py`
**Fixed critical merge issue**:
- Original version only merged global features (load, solar, wind)
- **Updated** to properly merge settlement point prices (DAM + RTM)
- Now creates correct grain: one row per hour per settlement point
- Enables DART target calculation per settlement point

---

## 🎯 Key Features Implemented

### Data Loading
- ✅ Loads parquet from Azure ML workspaceblobstore
- ✅ Automatic detection of feature types (categorical vs continuous)
- ✅ Time-based split (no data leakage)
- ✅ Target creation with missing value handling

### Feature Engineering
- ✅ **Categorical Encoding**: TargetEncoder for SettlementPoint
- ✅ **Standardization**: Z-score normalization (fit on train only)
- ✅ **Missing Value Handling**: Fill with 0 after standardization
- ✅ Proper train/val/test transformation pipeline

### Model Training
- ✅ **LightGBM**: Optimized hyperparameters, early stopping
- ✅ **XGBoost**: Tree-based regression, early stopping
- ✅ **Deep Learning**: LSTM with dropout, GPU support
- ✅ All models save with complete metadata

### Evaluation
- ✅ RMSE, MAE, MAPE, R² on all three sets
- ✅ Feature importance analysis (LightGBM, XGBoost)
- ✅ MLflow integration for experiment tracking

### Azure ML Integration
- ✅ Environment variable handling (`AZUREML_INPUT_*`, `AZUREML_OUTPUT_*`)
- ✅ Proper input/output path configuration
- ✅ Parallel pipeline execution
- ✅ Separate compute for CPU vs GPU workloads

---

## 📊 Data Pipeline Flow

```
Step 1: Feature Engineering
┌─────────────────────────────────────────┐
│ SQL Server (9 ERCOT Tables)             │
│ - Load tables (forecast + weather zones)│
│ - Price tables (DAM + RTM)              │
│ - Solar/Wind generation & forecasts     │
│ - SCED constraints                      │
└───────────────┬─────────────────────────┘
                ↓
        build_features.py
                ↓
┌─────────────────────────────────────────┐
│ hourly_features.parquet                 │
│ - TimestampHour (datetime)              │
│ - SettlementPoint (categorical)         │
│ - DAM_Price_Hourly (continuous)         │
│ - RTM_LMP_HourlyAvg (continuous)        │
│ - 50+ engineered features               │
└───────────────┬─────────────────────────┘
                ↓
Step 2: Model Training (Parallel)
┌─────────────────────────────────────────┐
│         dataloader.py                   │
│ 1. Load parquet                         │
│ 2. Create DART target                   │
│ 3. Time-based split (80/10/10)          │
│ 4. Encode SettlementPoint               │
│ 5. Standardize continuous features      │
└───┬─────────────────┬──────────────┬────┘
    ↓                 ↓              ↓
train_lgbm.py   train_xgb.py   train_deep.py
    ↓                 ↓              ↓
lgbm_model.pkl  xgb_model.pkl  deep_model.pt
```

---

## 🚀 How to Run

### Option 1: Individual Jobs

```bash
# Step 1: Build features
az ml job create --file aml_build_features.yml --web

# Step 2a: Train LightGBM
az ml job create --file aml_train_lgbm.yml --web

# Step 2b: Train XGBoost
az ml job create --file aml_train_xgb.yml --web

# Step 2c: Train Deep Learning
az ml job create --file aml_train_deep.yml --web
```

### Option 2: Full Pipeline (Recommended)

```bash
# Step 1: Build features
az ml job create --file aml_build_features.yml --web

# Wait for completion, then...

# Step 2: Train all models in parallel
az ml job create --file aml_training_pipeline.yml --web
```

---

## 📈 Expected Results

### Dataset Statistics
- **Rows**: ~1,000,000+ (hours × ~1053 settlement points)
- **Features**: ~50+ engineered features
- **Target**: DART spread ($/MWh)
- **Train**: 80% earliest data
- **Val**: 10% middle period
- **Test**: 10% most recent data

### Model Performance (estimated)
Based on typical DART spread prediction:
- **LightGBM**: RMSE ~$3-5/MWh, R² ~0.70-0.85
- **XGBoost**: RMSE ~$3-5/MWh, R² ~0.70-0.85
- **Deep Learning**: RMSE ~$4-6/MWh, R² ~0.65-0.80

(Actual performance depends on data quality and time period)

---

## 🔧 Customization Points

### Hyperparameter Tuning
All hyperparameters are defined in the training scripts:
- `train_lgbm.py`: lines 30-42 (num_leaves, learning_rate, etc.)
- `train_xgb.py`: lines 30-43 (max_depth, learning_rate, etc.)
- `train_deep.py`: lines 130-140 (hidden_dim, num_layers, dropout)

### Train/Val/Test Split
Modify `dataloader.py` line 83:
```python
train_pct=0.8, val_pct=0.1, test_pct=0.1
```

### Categorical Encoding Method
Change in `dataloader.py` line 138:
- Current: `TargetEncoder` (smooth mean encoding)
- Alternative: `OrdinalEncoder`, `CatBoostEncoder`

### Feature Selection
Modify `identify_feature_columns()` in `dataloader.py` to:
- Exclude specific features
- Add custom feature transformations
- Filter by importance

---

## ✅ Validation Checklist

- [x] All Python scripts have proper error handling
- [x] Azure ML environment variables are handled correctly
- [x] Time-based split prevents data leakage
- [x] Categorical encoding fits on train set only
- [x] Standardization fits on train set only
- [x] All models save with complete metadata
- [x] Metrics are computed for train/val/test
- [x] Feature importance is logged
- [x] GPU support for deep learning
- [x] Early stopping prevents overfitting
- [x] Missing values are handled properly
- [x] Target variable (DART) is correctly calculated

---

## 📝 Notes

### Critical Fix Applied
The original `build_features.py` had a merge issue where settlement point prices (DAM and RTM) were not included in the final output. This has been **fixed** - the merge now:
1. Starts with DAM prices (defines grain)
2. Merges RTM prices on (TimestampHour, SettlementPoint)
3. Broadcasts global features (load, solar, wind) on TimestampHour

This ensures the DART target can be calculated correctly.

### Compute Requirements
- **Feature Engineering**: 4-8 cores, 16-32 GB RAM, ~30-60 min
- **LightGBM/XGBoost**: 4-8 cores, 8-16 GB RAM, ~10-20 min each
- **Deep Learning**: 1-2 GPUs (V100/A100), 16-32 GB VRAM, ~30-60 min

### Storage
- **Input**: `hourly_features.parquet` (~1-5 GB)
- **Output Models**: 
  - LightGBM: ~50-200 MB
  - XGBoost: ~50-200 MB
  - Deep Learning: ~10-50 MB

---

## 🎓 Next Steps

1. **Run Feature Engineering**: Submit `aml_build_features.yml`
2. **Verify Output**: Check that `hourly_features.parquet` exists in workspaceblobstore
3. **Train Models**: Submit `aml_training_pipeline.yml`
4. **Compare Results**: Review metrics for all three models
5. **Select Best Model**: Based on test set performance
6. **Deploy**: Register and deploy the best model to an endpoint

---

## 📞 Support

See `PIPELINE_GUIDE.md` for detailed troubleshooting and customization options.

