# ERCOT ML Pipeline - Complete Project Structure

## 📁 File Organization

```
forecasting-ml/
│
├── 📊 STEP 1: FEATURE ENGINEERING
│   ├── build_features.py                 ✅ Main feature engineering script
│   └── aml_build_features.yml            ✅ Azure ML job definition
│
├── 🤖 STEP 2: MODEL TRAINING
│   │
│   ├── 📂 Core Training Scripts
│   │   ├── dataloader.py                 ✅ Data loading & preprocessing
│   │   ├── metrics.py                    ✅ Evaluation metrics (RMSE, MAE, MAPE, R²)
│   │   ├── train_lgbm.py                 ✅ LightGBM training
│   │   ├── train_xgb.py                  ✅ XGBoost training
│   │   └── train_deep.py                 ✅ Deep Learning (LSTM) training
│   │
│   ├── 📂 Azure ML Jobs (Individual)
│   │   ├── aml_train_lgbm.yml            ✅ LightGBM job (cpu-cluster)
│   │   ├── aml_train_xgb.yml             ✅ XGBoost job (cpu-cluster)
│   │   └── aml_train_deep.yml            ✅ Deep Learning job (gpu-cluster)
│   │
│   └── 📂 Azure ML Pipeline
│       └── aml_training_pipeline.yml     ✅ Parallel training pipeline
│
├── 🔧 CONFIGURATION
│   ├── requirements.txt                  ✅ Python dependencies
│   ├── environment.yml                   ✅ Conda environment (updated)
│   └── .env                              ⚠️  Create this (SQL credentials)
│
├── 📚 DOCUMENTATION
│   ├── README.md                         📖 Original project README
│   ├── PIPELINE_GUIDE.md                 ✅ Complete usage guide
│   ├── STEP2_SUMMARY.md                  ✅ Implementation summary
│   └── PROJECT_STRUCTURE.md              ✅ This file
│
└── 📂 OUTPUT DIRECTORIES (created automatically)
    ├── data/features/                    → hourly_features.parquet
    └── outputs/                          → trained models

```

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SQL SERVER                              │
│                      (9 ERCOT Tables)                           │
│                                                                 │
│  • hist_ActualSystemLoadbyForecastZone                         │
│  • hist_ActualSystemLoadbyWeatherZone                          │
│  • hist_DAMSettlementPointPrices                               │
│  • hist_LMPbyResourceNodesLoadZonesandTradingHubs              │
│  • hist_RealTimeLMP                                            │
│  • hist_SCEDShadowPricesandBindingTransmissionConstraints      │
│  • hist_SolarPowerProductionActual5MinuteAveragedValues        │
│  • hist_SolarPowerProductionHourlyAveragedActual...            │
│  • hist_WindPowerProductionHourlyAveragedActual...             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ pyodbc connection
                         ↓
         ╔═══════════════════════════════════════╗
         ║    build_features.py                  ║
         ║                                       ║
         ║  1. Load tables (chunked)             ║
         ║  2. Normalize timestamps              ║
         ║  3. Resample 5-min → hourly           ║
         ║  4. Melt wide → long                  ║
         ║  5. Merge all features                ║
         ║  6. Save to parquet                   ║
         ╚═══════════════╦═══════════════════════╝
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              AZURE ML WORKSPACEBLOBSTORE                        │
│         features/hourly_features.parquet                        │
│                                                                 │
│  Columns:                                                       │
│    • TimestampHour (datetime)                                  │
│    • SettlementPoint (categorical, ~1053 values)               │
│    • DAM_Price_Hourly ($/MWh)                                  │
│    • RTM_LMP_HourlyAvg ($/MWh)                                 │
│    • Load_NORTH_Hourly, Load_SOUTH_Hourly, ...                 │
│    • Solar_Actual_Hourly, Solar_Forecast_STPPF_Hourly, ...     │
│    • Wind_Actual_System_Hourly, Wind_Forecast_STWPF_*, ...     │
│    • 50+ total features                                        │
│                                                                 │
│  Rows: ~1M+ (hours × settlement points)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Read parquet
                         ↓
         ╔═══════════════════════════════════════╗
         ║    dataloader.py                      ║
         ║                                       ║
         ║  1. Load parquet                      ║
         ║  2. Create DART target                ║
         ║     DART = DAM - RTM                  ║
         ║  3. Time-based split                  ║
         ║     Train: 80% | Val: 10% | Test: 10%║
         ║  4. Encode SettlementPoint            ║
         ║  5. Standardize features              ║
         ╚═══════╦═══════════╦═══════════╦═══════╝
                 │           │           │
       ┌─────────┴───┐   ┌───┴─────┐   ┌┴────────────┐
       │             │   │         │   │             │
       ↓             ↓   ↓         ↓   ↓             ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ LightGBM     │ │ XGBoost      │ │ Deep Learning    │
│              │ │              │ │ (LSTM)           │
│ • Gradient   │ │ • Extreme    │ │ • 2-layer LSTM   │
│   boosting   │ │   gradient   │ │ • 128 hidden dim │
│ • 1000 trees │ │   boosting   │ │ • Dropout 0.2    │
│ • Early stop │ │ • Early stop │ │ • Early stop     │
│              │ │              │ │                  │
│ cpu-cluster  │ │ cpu-cluster  │ │ gpu-cluster      │
└──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
       │                │                  │
       ↓                ↓                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              AZURE ML WORKSPACEBLOBSTORE                        │
│                    models/                                      │
│                                                                 │
│  • lgbm/lgbm_model.pkl     (50-200 MB)                         │
│  • xgb/xgb_model.pkl       (50-200 MB)                         │
│  • deep/deep_model.pt      (10-50 MB)                          │
│                                                                 │
│  Each model includes:                                          │
│    - Trained model weights                                     │
│    - Feature column names                                      │
│    - StandardScaler (fitted on train)                          │
│    - Categorical encoders                                      │
│    - Train/Val/Test metrics                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Target Variable

```python
DART_Spread = DAM_Price_Hourly - RTM_LMP_HourlyAvg

# Where:
#   DAM_Price_Hourly     = Day-Ahead Market clearing price
#   RTM_LMP_HourlyAvg    = Real-Time Market LMP (avg of 5-min values)
```

**Why DART matters**:
- Measures market forecasting accuracy
- Indicates supply/demand imbalances
- Key metric for energy traders and operators
- Typical range: -$50 to +$50 per MWh
- Goal: Predict with RMSE < $5/MWh

---

## 📊 Feature Categories

### Load Features (13)
From forecast zones + weather zones:
- `Load_NORTH_Hourly`, `Load_SOUTH_Hourly`, `Load_WEST_Hourly`
- `Load_HOUSTON_Hourly`, `Load_TOTAL_Hourly`
- `Load_COAST_Hourly`, `Load_EAST_Hourly`, etc.

### Solar Features (9)
System-wide generation and forecasts:
- **Actual**: `Solar_Actual_Hourly`
- **Capacity**: `Solar_HSL_Hourly` (High Sustained Limit)
- **Forecasts**: `Solar_Forecast_STPPF_Hourly`, `Solar_Forecast_PVGRPP_Hourly`
- **COP**: `Solar_COP_HSL_Hourly` (Current Operating Plan)

### Wind Features (17)
System + 3 regional breakdowns:
- **System**: `Wind_Actual_System_Hourly`, `Wind_HSL_System_Hourly`
- **South Houston**: `Wind_Actual_SOUTH_HOUSTON_Hourly`, forecasts, HSL
- **West**: `Wind_Actual_WEST_Hourly`, forecasts, HSL
- **North**: `Wind_Actual_NORTH_Hourly`, forecasts, HSL

### Price Features (2)
- `DAM_Price_Hourly` → used to create target
- `RTM_LMP_HourlyAvg` → used to create target

### Categorical (1)
- `SettlementPoint` → Target-encoded (~1053 unique values)

---

## 🚀 Execution Commands

### Quick Start (Full Pipeline)

```bash
# 1. Build features (30-60 min)
az ml job create --file aml_build_features.yml --web

# 2. Train all models in parallel (30-60 min)
az ml job create --file aml_training_pipeline.yml --web
```

### Individual Model Training

```bash
# Train LightGBM only (10-20 min)
az ml job create --file aml_train_lgbm.yml --web

# Train XGBoost only (10-20 min)
az ml job create --file aml_train_xgb.yml --web

# Train Deep Learning only (30-60 min with GPU)
az ml job create --file aml_train_deep.yml --web
```

### Local Testing (requires .env file)

```bash
# Test feature engineering locally
python build_features.py

# Test model training locally (after features are built)
python train_lgbm.py
python train_xgb.py
python train_deep.py
```

---

## 🔐 Required Credentials

Create `.env` file in project root:

```env
SQL_SERVER=your-server.database.windows.net
SQL_DATABASE=ERCOT
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
```

---

## ⚙️ Compute Requirements

| Job Type          | Compute       | Cores | RAM    | GPU  | Time      |
|-------------------|---------------|-------|--------|------|-----------|
| Feature Build     | cpu-cluster   | 4-8   | 16-32G | No   | 30-60 min |
| LightGBM Train    | cpu-cluster   | 4-8   | 8-16G  | No   | 10-20 min |
| XGBoost Train     | cpu-cluster   | 4-8   | 8-16G  | No   | 10-20 min |
| Deep Learning     | gpu-cluster   | 4-8   | 16-32G | Yes  | 30-60 min |

---

## 📈 Expected Performance

Based on typical ERCOT DART spread prediction:

| Model         | RMSE ($/MWh) | MAE ($/MWh) | MAPE (%) | R²    |
|---------------|--------------|-------------|----------|-------|
| LightGBM      | 3.0 - 5.0    | 2.0 - 3.5   | 15 - 25  | 0.75+ |
| XGBoost       | 3.0 - 5.0    | 2.0 - 3.5   | 15 - 25  | 0.75+ |
| Deep Learning | 4.0 - 6.0    | 3.0 - 4.5   | 20 - 30  | 0.70+ |

*Actual results depend on data quality, time period, and hyperparameters*

---

## ✅ Completion Checklist

### Step 1: Feature Engineering
- [x] SQL connection handling
- [x] Chunked loading for large tables
- [x] Timestamp normalization (3 formats)
- [x] 5-minute → hourly resampling
- [x] Wide → long melting
- [x] Settlement point merge (FIXED)
- [x] Parquet output to Azure ML

### Step 2: Model Training
- [x] Data loader with time-based split
- [x] DART target creation
- [x] Categorical encoding (TargetEncoder)
- [x] Feature standardization
- [x] LightGBM training script
- [x] XGBoost training script
- [x] Deep Learning (LSTM) script
- [x] Evaluation metrics (RMSE, MAE, MAPE, R²)
- [x] Feature importance logging
- [x] Model serialization with metadata

### Azure ML Integration
- [x] Environment variable handling
- [x] Input/output path configuration
- [x] Individual job YAMLs (3)
- [x] Parallel pipeline YAML (1)
- [x] Compute cluster specification

### Documentation
- [x] Complete usage guide (PIPELINE_GUIDE.md)
- [x] Implementation summary (STEP2_SUMMARY.md)
- [x] Project structure (PROJECT_STRUCTURE.md)
- [x] Requirements file
- [x] Environment file

---

## 🎓 Learning Resources

### ERCOT Market Basics
- **DAM**: Day-Ahead Market (hourly auctions for next operating day)
- **RTM**: Real-Time Market (5-minute SCED dispatch)
- **DART**: Day-Ahead Real-Time spread (forecast error indicator)
- **LMP**: Locational Marginal Price (nodal pricing)
- **Settlement Points**: Nodes, hubs, load zones (~1053 total)

### Model Selection Guide
- **LightGBM**: Fast, memory-efficient, handles missing values
- **XGBoost**: Accurate, regularization built-in, slower than LightGBM
- **Deep Learning**: Captures complex patterns, requires more data/compute

---

## 📞 Troubleshooting

See `PIPELINE_GUIDE.md` section "Troubleshooting" for:
- SQL connection issues
- Azure ML path problems
- Missing dependencies
- GPU configuration
- Data quality issues

---

## 🎯 Next Steps

1. ✅ **Run Feature Engineering**: `az ml job create --file aml_build_features.yml`
2. ⏳ **Wait for Completion**: Check Azure ML Studio for job status
3. ✅ **Verify Parquet**: Ensure `hourly_features.parquet` exists in workspaceblobstore
4. ✅ **Run Training Pipeline**: `az ml job create --file aml_training_pipeline.yml`
5. 📊 **Compare Models**: Review metrics in Azure ML Studio
6. 🚀 **Deploy Best Model**: Register and create endpoint
7. 📈 **Monitor**: Set up drift detection and retraining schedule

---

**🎉 Pipeline Implementation Complete!**

All files are ready for execution. Follow the commands above to start training your ERCOT DART spread prediction models.

