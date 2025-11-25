# ERCOT ML Pipeline Verification Checklist

This checklist guides you through verifying the output of the feature engineering job and testing the input mounting for the training jobs.

**Feature Engineering Job ID:** `willing_hominy_5mkqggcb33` (This is the job that just started)
**Previous Job (had wrong path):** `dreamy_star_kxsnf8nzwg`

---

## ✅ **Phase 1: Wait for Feature Engineering to Complete (3-6 hours)**

### **Monitor Progress:**

```bash
# Check status every 30-60 minutes
az ml job show --name willing_hominy_5mkqggcb33 --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query "{Name:name, Status:status}" --output table
```

**Expected Timeline:**
- 0-15 min: Queued
- 15-30 min: Preparing (pulling Docker image)
- 30 min - 5 hours: Running (processing data)
- Final: Completed ✅

**Or check in Azure ML Studio:**
- https://ml.azure.com → **Jobs** → `willing_hominy_5mkqggcb33`

---

## ✅ **Phase 2: Verify Output Was Saved Correctly (5-10 minutes)**

Once the job shows **Status: Completed**, proceed with these steps:

### **Step 1: Check Job Logs for Output Path (2 minutes)**

1. Go to Azure ML Studio: https://ml.azure.com → **Jobs** → `willing_hominy_5mkqggcb33`
2. Click **"Outputs + logs"** tab
3. Open `user_logs/std_log.txt`
4. Search for **"✓ Using Azure ML output path"**

**Expected Output:**
```
✓ Using Azure ML output path from AZUREML_CR_OUTPUT_features: /mnt/azureml/.../features
✓ Saved features to: /mnt/azureml/.../features/hourly_features.parquet
```

**If you see this instead:**
```
⚠️ Azure ML output path not found, using local path
```
→ **STOP! The fix didn't work. Tell me immediately.**

---

### **Step 2: Download and Validate the Output (5 minutes)**

```bash
# Download the features output
az ml job download --name willing_hominy_5mkqggcb33 \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --download-path ./features_output \
  --output-name features

# Check if file exists and get size
if [ -f "./features_output/named-outputs/features/hourly_features.parquet" ]; then
    echo "✅ SUCCESS! File exists"
    ls -lh ./features_output/named-outputs/features/hourly_features.parquet | awk '{print "   Size: " $5}'
    echo "   Expected: ~186M"
else
    echo "❌ FAILED - File not found"
    find ./features_output -type f
fi
```

---

### **Step 3: Run Data Quality Validation (2 minutes)**

```bash
python validate_parquet.py --file ./features_output/named-outputs/features/hourly_features.parquet
```

**Expected Output:**
```
✅ ✅ ✅  ALL VALIDATION CHECKS PASSED  ✅ ✅ ✅
```

---

## ✅ **Phase 3: Test Input Mounting (2-3 minutes)**

Before submitting training jobs, let's verify Azure ML can mount the new output correctly.

### **Step 4: Update Test Job to Use New Output**

First, we need to update `aml_test_input_mount.yml` to point to the NEW job output.

**After the feature job completes**, find its output data asset name:

```bash
# This will be different from the old one, something like:
# azureml://datastores/workspaceblobstore/paths/azureml/<NEW_UUID>/features
```

Then update `aml_test_input_mount.yml` and submit:

```bash
az ml job create --file aml_test_input_mount.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

Wait ~2 minutes, then check the logs. **Expected:**
```
✅ ✅ ✅  ALL AZURE ML INPUT MOUNT TESTS PASSED  ✅ ✅ ✅
```

---

## ✅ **Phase 4: Submit Training Jobs (30-60 minutes total)**

Once Steps 1-4 all pass, update the training YAMLs and submit:

### **Step 5: Update Training YAMLs**

Update the `path` in all 3 files:
- `aml_train_lgbm.yml`
- `aml_train_xgb.yml`
- `aml_train_deep.yml`

Change from:
```yaml
path: azureml://datastores/workspaceblobstore/paths/azureml/bdd4515c-1b4d-4f60-a434-45e78df8f27e/features
```

To the NEW path (get the UUID from the completed job `willing_hominy_5mkqggcb33`).

### **Step 6: Submit All 3 Training Jobs**

```bash
# LightGBM
az ml job create --file aml_train_lgbm.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

# XGBoost
az ml job create --file aml_train_xgb.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

# Deep Learning
az ml job create --file aml_train_deep.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

**Expected Timeline:**
- LightGBM: ~15-20 minutes
- XGBoost: ~15-20 minutes
- Deep Learning: ~40-60 minutes

---

## 🎉 **Success Criteria**

All 3 training jobs should show:
- ✅ **Status: Completed**
- ✅ **"✓ Loaded X,XXX,XXX rows"** in logs (~23M rows)
- ✅ **"✓ Model saved to Azure ML"**
- ✅ **Final RMSE, MAE, R² metrics printed**

---

## 📞 **If Something Goes Wrong**

**Issue:** Feature engineering job fails
- Check `std_log.txt` for error
- Look for SQL connection issues or data type errors

**Issue:** Test job still shows empty directory
- The output path fix didn't work
- We need to debug the environment variables in `build_features.py`

**Issue:** Training jobs fail with "file not found"
- The path in the training YAMLs is incorrect
- Double-check the UUID matches the NEW feature job output

---

## ⏰ **Current Time Estimate**

| Phase | Duration | Status |
|-------|----------|--------|
| Feature Engineering | 3-6 hours | 🏃 **Running Now** |
| Verification | 10 min | ⏳ Waiting |
| Test Mount | 2 min | ⏳ Waiting |
| Training (all 3) | 60 min | ⏳ Waiting |
| **Total** | **4-7 hours** | |

---

**💡 Tip:** Set a reminder to check back in 4-5 hours, or monitor via Azure ML Studio.
