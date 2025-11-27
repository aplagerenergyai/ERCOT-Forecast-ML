# ERCOT ML Pipeline - Step 2: Model Training Guide

## 📋 Overview

This guide covers Step 2 of the ERCOT ML pipeline: training three models (LightGBM, XGBoost, Deep Learning) on the engineered features from Step 1.

---

## 🎯 Prerequisites

✅ **Step 1 Completed**: Feature engineering parquet file generated
- Located at: `azureml://datastores/workspaceblobstore/paths/features/hourly_features.parquet`
- Contains ~20M hourly records with 17 features
- Successfully passed the 245M row processing job

---

## 📦 New Files Created

### 1. **`run_training_jobs.sh`**
Bash script to submit all three training jobs to Azure ML in parallel.

**Features:**
- Submits LightGBM, XGBoost, and Deep Learning jobs simultaneously
- Captures job IDs and logs to `.azureml/training_jobs.log`
- Provides monitoring commands for each job
- Color-coded output for status

### 2. **`validate_parquet.py`**
Python script to validate the feature parquet file before training.

**Checks:**
- ✅ Row count and column count
- ✅ Null value analysis per column
- ✅ Timestamp continuity (hourly gaps)
- ✅ DART spread calculation (DAM - RTM prices)
- ✅ Feature distributions
- ✅ Settlement point analysis (if applicable)
- ✅ Memory usage and file size

### 3. **Updated Training YAMLs**
All three training job YAMLs updated to use:
- Custom Docker image: `ercotforecastingprod.azurecr.io/ercot-ml-pipeline:latest`
- Proper compute reference: `azureml:cpu-cluster`
- Display names for easy identification in Azure ML Studio

**Files:**
- `aml_train_lgbm.yml` - LightGBM training
- `aml_train_xgb.yml` - XGBoost training
- `aml_train_deep.yml` - Deep Learning (LSTM) training

### 4. **Updated Makefile**
Two new targets added:
- `make validate` - Validate parquet file locally
- `make train` - Submit all training jobs to Azure ML

---

## 🚀 Quick Start

### Option A: Using Makefile (Easiest)

```bash
# Step 1: Validate features (optional but recommended)
make validate

# Step 2: Submit all training jobs
make train
```

### Option B: Using Scripts Directly

```bash
# Step 1: Validate features
python validate_parquet.py --file path/to/hourly_features.parquet

# Step 2: Submit training jobs
bash run_training_jobs.sh
```

### Option C: Manual Azure ML Submission

```bash
# Submit individual jobs
az ml job create --file aml_train_lgbm.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
az ml job create --file aml_train_xgb.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
az ml job create --file aml_train_deep.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
```

---

## 📊 Validation Workflow

### Step 1: Download Features from Azure ML (if needed)

```bash
# Download the parquet file from the completed feature engineering job
JOB_NAME="funny_brain_3722xqqq7p"  # Your actual job name

az ml job download \
  --name $JOB_NAME \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./features_output \
  --output-name features
```

### Step 2: Run Validation

```bash
# Auto-detect parquet file in common locations
python validate_parquet.py

# Or specify explicit path
python validate_parquet.py --file ./features_output/features/hourly_features.parquet
```

### Expected Validation Output

```
================================================================================
  ERCOT FEATURE ENGINEERING - PARQUET VALIDATION
================================================================================

📁 File: ./features_output/features/hourly_features.parquet
💾 Size: 245.67 MB

================================================================================
  1. BASIC INFORMATION
================================================================================

📊 Shape: 20,147,964 rows × 17 columns
💾 Memory Usage: 2,456.78 MB
✅ PASS: DataFrame has data

================================================================================
  5. DART SPREAD VALIDATION
================================================================================

📊 Price Data Availability:
   DAM prices present: 18,234,567 rows (90.5%)
   RTM prices present: 19,456,789 rows (96.5%)

📈 DART Spread Statistics:
   Valid DART spread rows: 18,234,567 (90.5%)
   Mean: $2.34
   Std:  $5.67
   Min:  $-45.23
   Max:  $123.45

✅ PASS: DART spread available for 90.5% of rows

================================================================================
  VALIDATION SUMMARY
================================================================================

✅ ✅ ✅  ALL VALIDATION CHECKS PASSED  ✅ ✅ ✅

🎉 The parquet file is ready for model training!
```

---

## 🧠 Training Workflow

### Step 1: Submit Jobs

```bash
# Using Makefile (recommended)
make train

# Or using script directly
bash run_training_jobs.sh
```

### Expected Output

```
════════════════════════════════════════════════════════════════
  ERCOT ML Training Jobs - Azure ML Submission
════════════════════════════════════════════════════════════════

Configuration:
  Workspace: energyaiml-prod
  Resource Group: rg-ercot-ml-production
  Log File: .azureml/training_jobs.log

Submitting training jobs in parallel...

Submitting LightGBM Training...
✓ Submitted: happy_cloud_abc123xyz

Submitting XGBoost Training...
✓ Submitted: bright_sky_def456uvw

Submitting Deep Learning Training...
✓ Submitted: calm_river_ghi789rst

════════════════════════════════════════════════════════════════
  Submission Summary
════════════════════════════════════════════════════════════════

✓ LightGBM:      happy_cloud_abc123xyz
✓ XGBoost:       bright_sky_def456uvw
✓ Deep Learning: calm_river_ghi789rst

Monitor jobs with:
  az ml job show --name happy_cloud_abc123xyz --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
  az ml job show --name bright_sky_def456uvw --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
  az ml job show --name calm_river_ghi789rst --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production

Or visit Azure ML Studio:
  https://ml.azure.com

✓ All training jobs submitted successfully!
```

### Step 2: Monitor Jobs

#### Option A: Azure ML Studio (Visual)
1. Go to https://ml.azure.com
2. Navigate to **Jobs**
3. Look for:
   - `Train_LightGBM_DART_Spread`
   - `Train_XGBoost_DART_Spread`
   - `Train_DeepLearning_DART_Spread`

#### Option B: Azure CLI

```bash
# Check status of all jobs
JOB1="happy_cloud_abc123xyz"
JOB2="bright_sky_def456uvw"
JOB3="calm_river_ghi789rst"

az ml job show --name $JOB1 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query "{Name:display_name, Status:status}" -o table
az ml job show --name $JOB2 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query "{Name:display_name, Status:status}" -o table
az ml job show --name $JOB3 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query "{Name:display_name, Status:status}" -o table
```

#### Option C: Check Logs

```bash
# View job IDs from log file
cat .azureml/training_jobs.log

# Stream logs for a specific job
az ml job stream --name $JOB1 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production
```

---

## ⏱️ Expected Training Times

| Model | Compute | Expected Duration | Output |
|-------|---------|-------------------|--------|
| **LightGBM** | cpu-cluster (Standard_DS3_v2) | ~10-15 minutes | `models/lgbm/lgbm_model.pkl` |
| **XGBoost** | cpu-cluster (Standard_DS3_v2) | ~10-15 minutes | `models/xgb/xgb_model.pkl` |
| **Deep Learning** | cpu-cluster (Standard_DS3_v2) | ~30-45 minutes | `models/deep/deep_model.pt` |

**Total parallel time:** ~45 minutes (since they run simultaneously)

---

## 📦 Model Outputs

After training completes, models will be saved to:

```
azureml://datastores/workspaceblobstore/paths/models/
├── lgbm/
│   └── lgbm_model.pkl
├── xgb/
│   └── xgb_model.pkl
└── deep/
    └── deep_model.pt
```

### Download Trained Models

```bash
# Download LightGBM model
az ml job download \
  --name $JOB1 \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./models_output \
  --output-name model

# Download XGBoost model
az ml job download \
  --name $JOB2 \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./models_output \
  --output-name model

# Download Deep Learning model
az ml job download \
  --name $JOB3 \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./models_output \
  --output-name model
```

---

## 📊 Model Evaluation Metrics

Each training script outputs:

- **RMSE** (Root Mean Squared Error) - Lower is better
- **MAE** (Mean Absolute Error) - Lower is better
- **MAPE** (Mean Absolute Percentage Error) - Lower is better
- **R²** (R-squared) - Higher is better (max 1.0)

### Check Metrics in Logs

```bash
# View metrics from log file
az ml job download --name $JOB1 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --download-path ./job1_output
cat ./job1_output/user_logs/std_log.txt | grep -A 10 "Evaluation Metrics"
```

Expected output:
```
Evaluation Metrics:
  RMSE: 3.45
  MAE:  2.12
  MAPE: 15.3%
  R²:   0.87
```

---

## 🐛 Troubleshooting

### Issue 1: "Feature file not found"

**Symptom:** Training job fails with `FileNotFoundError: hourly_features.parquet`

**Solution:**
```bash
# Verify the feature file exists in workspaceblobstore
az storage blob list \
  --account-name energyaistorage04ae16f9c \
  --container-name azureml-blobstore-* \
  --prefix "features/" \
  --auth-mode login
```

### Issue 2: "Not enough data for DART target"

**Symptom:** Training fails with insufficient rows for target calculation

**Solution:**
- Re-run `validate_parquet.py` to check DART spread availability
- Ensure at least 50% of rows have both DAM and RTM prices
- Check the feature engineering logs for settlement point processing errors

### Issue 3: "Out of memory during training"

**Symptom:** Job fails with SIGKILL or OOM error

**Solution:**
- Training should work fine on `cpu-cluster` (14GB RAM)
- If deep learning fails, reduce batch size in `train_deep.py`
- Or temporarily use `memory-cluster` (128GB) for deep learning

### Issue 4: "Container image pull failed"

**Symptom:** Job fails to start with Docker image pull error

**Solution:**
```bash
# Verify ACR connection
az ml workspace show \
  --name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query container_registry

# Re-attach ACR if needed
az ml workspace update \
  --name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --container-registry /subscriptions/b7788659-1f79-4e40-b98a-eea87041561f/resourceGroups/rg-ercot-ml-production/providers/Microsoft.ContainerRegistry/registries/ercotforecastingprod \
  --update-dependent-resources
```

---

## ✅ Success Criteria

Training is successful when:

1. ✅ All three jobs complete with status "Completed"
2. ✅ Each job produces a model artifact in workspaceblobstore
3. ✅ Evaluation metrics are reasonable:
   - RMSE < 10
   - MAE < 5
   - R² > 0.70
4. ✅ No errors in job logs

---

## 🚀 Next Steps

After training completes successfully:

1. **Compare Model Performance**
   - Review metrics for all three models
   - Select the best performing model (likely LightGBM or XGBoost)

2. **Test Inference**
   - Deploy model to online endpoint
   - Test with sample data
   - Verify predictions are reasonable

3. **Step 3: Production Pipeline**
   - Create end-to-end pipeline (feature → train → predict)
   - Set up scheduling for daily/hourly runs
   - Add monitoring and alerting

---

## 📚 Additional Resources

- **Azure ML Documentation:** https://learn.microsoft.com/en-us/azure/machine-learning/
- **Training Scripts:** `train_lgbm.py`, `train_xgb.py`, `train_deep.py`
- **Data Loader:** `dataloader.py`
- **Metrics Helper:** `metrics.py`

---

## 📞 Support

If you encounter issues:
1. Check job logs in Azure ML Studio
2. Review `.azureml/training_jobs.log` for submission details
3. Re-run validation: `make validate`
4. Check this guide's troubleshooting section

---

**Ready to train? Run `make train` and monitor the jobs!** 🚀

