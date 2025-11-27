# ERCOT ML Pipeline - Quick Start Guide

Complete end-to-end execution guide for the ERCOT DART spread prediction pipeline.

---

## 🚀 Complete Pipeline Execution (3 Steps)

### Step 1: Feature Engineering (30-60 min)

Build the unified hourly features from SQL Server:

```bash
az ml job create --file aml_build_features.yml --web
```

**What it does**:
- Connects to SQL Server
- Loads 9 ERCOT tables
- Normalizes timestamps
- Resamples 5-min data to hourly
- Merges all features
- Saves `hourly_features.parquet` to workspaceblobstore

**Output**: `workspaceblobstore/paths/features/hourly_features.parquet`

---

### Step 2: Model Training (30-60 min parallel)

Train all three models in parallel:

```bash
az ml job create --file aml_training_pipeline.yml --web
```

**What it does**:
- Loads features from Step 1
- Splits data (80/10/10 time-based)
- Creates DART target (DAM - RTM)
- Trains LightGBM, XGBoost, and LSTM in parallel
- Evaluates on train/val/test
- Saves all three models

**Outputs**:
- `workspaceblobstore/paths/models/lgbm/lgbm_model.pkl`
- `workspaceblobstore/paths/models/xgb/xgb_model.pkl`
- `workspaceblobstore/paths/models/deep/deep_model.pt`

---

### Step 3: Model Comparison & Deployment

Download and compare models:

```bash
# Download all model outputs
az ml job download --name <pipeline-job-name> --all

# View metrics
python compare_models.py
```

---

## 📋 Alternative: Run Individual Components

### Build Features Only
```bash
az ml job create --file aml_build_features.yml --web
```

### Train Individual Models

```bash
# LightGBM only
az ml job create --file aml_train_lgbm.yml --web

# XGBoost only
az ml job create --file aml_train_xgb.yml --web

# Deep Learning only
az ml job create --file aml_train_deep.yml --web
```

---

## 🔍 Monitoring & Status

### Check Job Status
```bash
az ml job show --name <job-name>
```

### Stream Job Logs
```bash
az ml job stream --name <job-name>
```

### List Recent Jobs
```bash
az ml job list --max-results 10
```

### View in Azure Portal
Navigate to: **Azure ML Studio → Jobs → [Your Job]**

---

## 📊 Expected Timeline

| Step | Job Type | Duration | Can Run in Parallel? |
|------|----------|----------|----------------------|
| 1 | Feature Build | 30-60 min | No (single job) |
| 2a | LightGBM Train | 10-20 min | Yes |
| 2b | XGBoost Train | 10-20 min | Yes |
| 2c | Deep Learning | 30-60 min | Yes |
| **Total** | **End-to-End** | **~60-120 min** | **Step 2 jobs run together** |

---

## ✅ Prerequisites Checklist

Before running the pipeline:

- [ ] `.env` file created with SQL credentials
  ```env
  SQL_SERVER=your-server.database.windows.net
  SQL_DATABASE=ERCOT
  SQL_USERNAME=your-username
  SQL_PASSWORD=your-password
  ```

- [ ] Azure ML workspace exists
- [ ] Compute clusters created:
  - [ ] `cpu-cluster` (Standard_D4s_v3 or similar)
  - [ ] `gpu-cluster` (Standard_NC6 or similar)
- [ ] Azure CLI installed and logged in:
  ```bash
  az login
  az account set --subscription <subscription-id>
  ```
- [ ] Required Python packages installed:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📁 Output Structure

After complete pipeline execution:

```
workspaceblobstore/
├── features/
│   └── hourly_features.parquet         (Step 1 output)
│
└── models/
    ├── lgbm/
    │   └── lgbm_model.pkl              (Step 2a output)
    ├── xgb/
    │   └── xgb_model.pkl               (Step 2b output)
    └── deep/
        └── deep_model.pt               (Step 2c output)
```

---

## 🛠️ Troubleshooting

### Job Fails Immediately
**Check**:
- Compute cluster exists and is available
- Environment has all required packages
- Input data paths are correct

**Solution**:
```bash
# View detailed error
az ml job show --name <job-name> --query error
```

### Feature Building Fails
**Likely causes**:
- SQL Server connection issue
- Missing .env file
- ODBC driver not installed

**Solution**:
- Verify `.env` credentials
- Test SQL connection locally
- Check firewall rules

### Training Fails with "Input not found"
**Cause**: Features not built yet

**Solution**:
```bash
# Run Step 1 first
az ml job create --file aml_build_features.yml --web
# Wait for completion, then run Step 2
```

### GPU Cluster Not Found
**Cause**: GPU cluster doesn't exist

**Solution**: Either create GPU cluster or modify `aml_train_deep.yml`:
```yaml
compute: cpu-cluster  # Use CPU instead (slower)
```

---

## 🎯 Success Criteria

### After Step 1 (Feature Engineering)
✅ Job status: Completed  
✅ File exists: `workspaceblobstore/paths/features/hourly_features.parquet`  
✅ File size: 1-5 GB (typical)  
✅ Rows: ~1M+ (hours × settlement points)  

### After Step 2 (Model Training)
✅ Pipeline status: Completed  
✅ All 3 jobs succeeded  
✅ Models saved:
  - `lgbm_model.pkl` (50-200 MB)
  - `xgb_model.pkl` (50-200 MB)
  - `deep_model.pt` (10-50 MB)  
✅ Test RMSE: < $10/MWh (ideally < $5/MWh)  
✅ Test R²: > 0.65 (ideally > 0.75)  

---

## 📈 Model Performance Expectations

Based on typical ERCOT DART spread prediction:

| Metric | LightGBM | XGBoost | Deep Learning |
|--------|----------|---------|---------------|
| RMSE ($/MWh) | 3-5 | 3-5 | 4-6 |
| MAE ($/MWh) | 2-3.5 | 2-3.5 | 3-4.5 |
| MAPE (%) | 15-25 | 15-25 | 20-30 |
| R² | 0.75-0.85 | 0.75-0.85 | 0.70-0.80 |
| Training Time | 10-20 min | 10-20 min | 30-60 min |

*Best performer varies by dataset and time period*

---

## 🔄 Re-running the Pipeline

### Update Features with New Data
```bash
# Re-run feature engineering
az ml job create --file aml_build_features.yml --web

# Re-train all models with new features
az ml job create --file aml_training_pipeline.yml --web
```

### Re-train Specific Model
```bash
# Only re-train LightGBM
az ml job create --file aml_train_lgbm.yml --web
```

---

## 💡 Tips for Better Performance

### Hyperparameter Tuning
Modify training scripts to adjust hyperparameters:
- `train_lgbm.py` line 30-42
- `train_xgb.py` line 30-43
- `train_deep.py` line 140-145

### Feature Selection
Edit `dataloader.py` to:
- Remove low-importance features
- Add feature engineering transformations
- Adjust categorical encoding method

### Data Split Optimization
Modify split ratio in `dataloader.py` line 83:
```python
train_pct=0.8, val_pct=0.1, test_pct=0.1
```

---

## 🎓 Next Steps After Training

1. **Compare Models**: Review metrics for all three models
2. **Select Best**: Choose based on test RMSE and business requirements
3. **Register Model**: Add to Azure ML model registry
4. **Deploy**: Create real-time or batch endpoint
5. **Monitor**: Set up data drift and performance monitoring
6. **Schedule Retraining**: Automate monthly/quarterly updates

---

## 📞 Getting Help

- **Azure ML Documentation**: https://docs.microsoft.com/azure/machine-learning/
- **ERCOT Data Guide**: http://www.ercot.com/
- **Project Documentation**:
  - `PIPELINE_GUIDE.md` - Complete usage guide
  - `STEP3_PIPELINE.md` - Pipeline architecture details
  - `PROJECT_STRUCTURE.md` - File organization

---

## 🎉 Quick Command Summary

```bash
# Complete pipeline (2 commands)
az ml job create --file aml_build_features.yml --web
az ml job create --file aml_training_pipeline.yml --web

# That's it! ✨
```

---

**Ready to start?** Run the first command above! 🚀

