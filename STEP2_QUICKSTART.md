# Step 2: Model Training - Quick Start

## 🎯 **3-Step Quick Start**

### **Step 1: Download & Validate Features**
```bash
# Download from Azure ML (use your actual job name)
JOB_NAME="funny_brain_3722xqqq7p"
az ml job download \
  --name $JOB_NAME \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./features_output \
  --output-name features

# Validate
make validate
# or: python validate_parquet.py
```

### **Step 2: Submit Training Jobs**
```bash
# Submit all 3 models in parallel
make train
# or: bash run_training_jobs.sh
```

### **Step 3: Monitor Progress**
Visit: https://ml.azure.com → Jobs

Or use CLI:
```bash
# Check status (get job names from Step 2 output)
az ml job show --name <JOB_NAME> --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query status
```

---

## ⏱️ **Timeline**

| Step | Time |
|------|------|
| Validation | 2-3 minutes |
| Job submission | 1 minute |
| Training (parallel) | ~45 minutes |
| **Total** | **~50 minutes** |

---

## ✅ **Success Indicators**

You'll know it worked when:
- ✅ `make validate` outputs: "ALL VALIDATION CHECKS PASSED"
- ✅ `make train` outputs: "All training jobs submitted successfully"
- ✅ Azure ML Studio shows all 3 jobs as "Completed" (green)
- ✅ Models appear in `workspaceblobstore/paths/models/`

---

## 📦 **What Gets Created**

### Inputs (from Step 1):
- `hourly_features.parquet` (20M rows, 17 features)

### Outputs (from Step 2):
- `models/lgbm/lgbm_model.pkl` - LightGBM trained model
- `models/xgb/xgb_model.pkl` - XGBoost trained model
- `models/deep/deep_model.pt` - Deep Learning (PyTorch) trained model

---

## 🚨 **Common Issues & Quick Fixes**

### Issue: "Parquet file not found"
```bash
# Re-download from Azure ML
az ml job download --name <JOB_NAME> --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --download-path ./features_output --output-name features
```

### Issue: "Job submission failed"
```bash
# Check Azure CLI is logged in
az account show

# If not logged in:
az login
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f
```

### Issue: "Training job failed"
```bash
# Download logs
az ml job download --name <FAILED_JOB_NAME> --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --download-path ./failed_job_output

# Check error in logs
cat ./failed_job_output/user_logs/std_log.txt
```

---

## 📊 **Expected Metrics**

Good models will have:
- **RMSE**: 2-5 (dollars)
- **MAE**: 1.5-3 (dollars)
- **R²**: 0.75-0.90 (higher is better)

If metrics are worse:
- Check DART spread availability in validation output
- Verify feature quality (no excessive nulls)
- Consider adding more features or tuning hyperparameters

---

## 🎯 **Ready to Go?**

```bash
# One-liner to validate and train
make validate && make train
```

Then monitor at: **https://ml.azure.com** 🚀

---

For detailed information, see: **STEP2_TRAINING_GUIDE.md**

