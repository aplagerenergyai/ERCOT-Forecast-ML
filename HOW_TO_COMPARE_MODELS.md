# 🏆 How to Compare Models

## 📋 **Quick Start:**

```bash
python compare_models.py
```

---

## 📊 **What It Does:**

The `compare_models.py` script:
1. ✅ Queries Azure ML for all completed DART jobs
2. ✅ Reads metrics from the `manual_results` dictionary
3. ✅ Prints a comparison table sorted by Test MAE (best first)
4. ✅ Identifies and announces the winner
5. ✅ Calculates improvement over baseline

---

## 🔧 **How to Use:**

### **Step 1: Wait for Jobs to Complete**

Check job status at: https://ml.azure.com

Or run:
```bash
az ml job list --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query "[?contains(display_name, 'DART')].{Name:name, DisplayName:display_name, Status:status}" \
  --output table
```

---

### **Step 2: Get Metrics from Each Job**

For each completed job:

1. Go to https://ml.azure.com
2. Click the job name
3. Click **"Outputs + logs"** tab
4. Navigate to: `user_logs/std_log.txt`
5. Search for: **"Test Set Metrics:"**

You'll find:
```
Test Set Metrics:
  RMSE: $124.05
  MAE:  $11.90     ← Copy this
  MAPE: 291.45%    ← And this
  R²:   0.0001     ← And this
```

---

### **Step 3: Update the Script**

Open `compare_models.py` and update the `manual_results` dictionary:

```python
manual_results = {
    'Train_LightGBM_DART_Spread': {'mae': 11.90, 'rmse': 124.05, 'r2': 0.0001, 'mape': 291.45},
    'Train_XGBoost_DART_Spread': {'mae': 12.83, 'rmse': 125.20, 'r2': -0.0015, 'mape': 295.30},
    
    # Add your results here as jobs complete:
    'Train_CatBoost_DART_Spread': {'mae': 10.50, 'rmse': 120.00, 'r2': 0.05, 'mape': 250.00},
    'Train_RandomForest_DART_Spread': {'mae': 11.50, 'rmse': 122.00, 'r2': 0.02, 'mape': 280.00},
    # ... etc
}
```

---

### **Step 4: Run the Comparison**

```bash
python compare_models.py
```

---

## 📊 **Expected Output:**

```
====================================================================================================
  🔍 ERCOT ML MODEL COMPARISON TOOL
====================================================================================================

  Timestamp: 2025-11-26 15:30:00
  Workspace: energyaiml-prod
  Resource Group: rg-ercot-ml-production

✅ Found 8 completed DART jobs

====================================================================================================
  🏆 ERCOT DART MODEL COMPARISON
====================================================================================================

Model                                    Test MAE        Test RMSE       Test R²      Status    
----------------------------------------------------------------------------------------------------
Train_AutoML_DART_Spread                 $9.80           $118.00         0.0800       ✅ 🥇
Train_CatBoost_DART_Spread               $10.50          $120.00         0.0500       ✅ 🥈
Train_Ensemble_DART_Spread               $10.75          $121.00         0.0450       ✅ 🥉
Train_RandomForest_DART_Spread           $11.50          $122.00         0.0200       ✅
Train_LightGBM_DART_Spread               $11.90          $124.05         0.0001       ✅
Train_XGBoost_DART_Spread                $12.83          $125.20         -0.0015      ✅
----------------------------------------------------------------------------------------------------

====================================================================================================
  🏆 WINNER - BEST MODEL
====================================================================================================

  Model: Train_AutoML_DART_Spread
  Test MAE:  $9.80  ⭐ LOWEST ERROR
  Test RMSE: $118.00
  Test R²:   0.0800
  Test MAPE: 240.50%

====================================================================================================

  🎉 17.6% improvement over LightGBM baseline ($11.90)
  💰 Average error reduced by $2.10 per prediction

====================================================================================================
  📊 SUMMARY STATISTICS
====================================================================================================

  Models evaluated: 6
  Best MAE:   $9.80
  Worst MAE:  $12.83
  Average MAE: $11.21
  Range:      $3.03

```

---

## 🎯 **Ranking Criteria:**

The script ranks models by **Test MAE** (Mean Absolute Error):

- ✅ **Lower MAE = Better** (less error on average)
- 🥇 First place: Lowest MAE
- 🥈 Second place: Second lowest MAE
- 🥉 Third place: Third lowest MAE

---

## 📋 **Model Names to Use:**

Copy these exact names into `manual_results`:

```python
# Original 5 + Ensemble:
'Train_LightGBM_DART_Spread'
'Train_XGBoost_DART_Spread'
'Train_DeepLearning_DART_Spread'
'Train_CatBoost_DART_Spread'
'Train_RandomForest_DART_Spread'
'Train_Ensemble_DART_Spread'

# New 6 models:
'Train_AutoML_DART_Spread'
'Train_ExtraTrees_DART_Spread'
'Train_HistGradientBoosting_DART_Spread'
'Train_TabNet_DART_Spread'
'Train_NGBoost_DART_Spread'
'Train_TFT_DART_Spread'
```

---

## ⚠️ **Important Notes:**

1. **You must manually update metrics** - the script doesn't auto-download logs (Azure CLI limitation)
2. **Only completed jobs are compared** - running/failed jobs are ignored
3. **Baseline is LightGBM** at $11.90 MAE - improvements are calculated vs this
4. **Null values show as "Check logs"** - add those metrics to `manual_results`

---

## 🔄 **Workflow:**

```
1. Submit jobs → 2. Wait for completion → 3. Check logs at ml.azure.com
                                                    ↓
                                          4. Copy metrics to script
                                                    ↓
                                          5. Run compare_models.py
                                                    ↓
                                          6. See winner! 🏆
```

---

## 📞 **Need Help?**

If the script doesn't work:
1. Make sure Azure CLI is installed: `az --version`
2. Make sure you're logged in: `az login`
3. Check the workspace/resource group names are correct
4. Verify you have access to the ML workspace

---

**Happy model comparison!** 🚀

