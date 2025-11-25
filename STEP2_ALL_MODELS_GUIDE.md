# STEP 2: Train ALL Models - Complete Guide

## 🎯 **What We're Training**

You now have **7 different models** ready to train:

| # | Model | Status | Priority | Expected MAE | Time |
|---|-------|--------|----------|--------------|------|
| 1 | **LightGBM** | ✅ **DONE** | High | $11.90 | ~3 min |
| 2 | **XGBoost** | ✅ **DONE** | High | $12.83 | ~3 min |
| 3 | **Deep Learning (LSTM)** | 🏃 Running | Medium | $10-14 | ~60 min |
| 4 | **CatBoost** | ⏳ Ready | **HIGHEST** | **$10-11** | ~30 min |
| 5 | **Random Forest** | ⏳ Ready | High | $11-12 | ~40 min |
| 6 | **Ensemble** | ⏳ Ready | Very High | **$10-11** | ~5 min |
| 7 | **Azure AutoML** | ⏳ Ready | Ultra High | **$9-10** | ~2 hrs |

---

## 🚀 **Quick Start - Run Everything**

### **Option A: Run Models One By One (Recommended First Time)**

```bash
# 1. CatBoost (Best categorical handling)
az ml job create --file aml_train_catboost.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

# Wait ~30 minutes, then:

# 2. Random Forest (Most robust)
az ml job create --file aml_train_random_forest.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

# Wait ~40 minutes, then:

# 3. Ensemble (Combines all models)
az ml job create --file aml_train_ensemble.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

---

### **Option B: Run Automated Script (Submits & Waits)**

```bash
# This script submits each model and waits for it to complete
bash run_all_models.sh
```

---

## 📊 **Model Details**

### **1. CatBoost** ⭐⭐⭐⭐⭐

**Why it's best:**
- Superior handling of your 1,003 settlement points (categorical feature)
- Less prone to overfitting than XGBoost
- Often beats LightGBM/XGBoost on real-world data

**Key Features:**
- Native categorical support (no encoding needed)
- Ordered boosting (reduces overfitting)
- GPU support (if available)

**Expected Performance:** $10-11 MAE (better than current $11.90)

---

### **2. Random Forest** ⭐⭐⭐⭐

**Why it's robust:**
- More resistant to extreme outliers (your data has $-27,868 to $3,694 range!)
- Parallel tree building (uses all CPU cores efficiently)
- Less hyperparameter tuning needed

**Key Features:**
- 200 decision trees
- Bootstrap aggregating (bagging)
- Built-in cross-validation

**Expected Performance:** $11-13 MAE (similar to current)

---

### **3. Ensemble** ⭐⭐⭐⭐⭐

**Why it's powerful:**
- Combines strengths of all models
- Automatically weights models by validation performance
- Typically 5-10% better than best individual model

**Requirements:**
- ✅ LightGBM trained
- ✅ XGBoost trained
- ⏳ CatBoost trained (run this first)
- ⏳ Random Forest trained (run this second)
- ⏳ Deep Learning trained (optional)

**Expected Performance:** $10-11 MAE (best of all)

---

### **4. Azure AutoML** ⭐⭐⭐⭐⭐

**Why it's ultimate:**
- Tries 20+ algorithms automatically
- Hyperparameter tuning included
- Feature engineering built-in
- Model explainability included

**Algorithms AutoML Will Try:**
- All boosting variants (LightGBM, XGBoost tuned)
- Ensemble methods (Voting, Stacking)
- Linear models (ElasticNet, LARS)
- Tree ensembles (RandomForest, ExtraTrees)
- Nearest neighbors (KNN)
- And more...

**Configuration:**
- 2-hour experiment timeout
- 5-fold cross-validation
- Early stopping enabled
- Stack & voting ensembles enabled

**Expected Performance:** $9-10 MAE (best possible)

**Note:** AutoML configuration coming in next message...

---

## 📋 **Execution Plan**

### **Phase 1: Core Models (Today - 2 hours total)**

1. ✅ **LightGBM** - DONE ($11.90 MAE)
2. ✅ **XGBoost** - DONE ($12.83 MAE)
3. 🏃 **Deep Learning** - Running (~60 min remaining)
4. ⏳ **CatBoost** - Submit now (~30 min)
5. ⏳ **Random Forest** - Submit after CatBoost (~40 min)
6. ⏳ **Ensemble** - Submit after RF (~5 min)

**Total Time:** ~2 hours (all in parallel or sequence)

---

### **Phase 2: AutoML (Optional - 2 hours)**

7. ⏳ **Azure AutoML** - Run overnight (~2 hours)

**Total Time:** 2 hours (can run overnight)

---

## 🎯 **Expected Results**

### **Current Best:**
- LightGBM: $11.90 MAE

### **After New Models:**
- CatBoost: $10-11 MAE (**10-15% better**)
- Ensemble: $10-11 MAE (**10-15% better**)
- AutoML: $9-10 MAE (**20-25% better**)

---

## 📊 **How to Compare Models**

After all jobs complete, check each job's `std_log.txt` for:

```
Test Set Metrics:
  RMSE: $XX.XX  ← Lower is better
  MAE:  $XX.XX  ← Lower is better (MAIN METRIC)
  R²:   X.XX    ← Higher is better
```

**Ranking Criteria:**
1. **Primary:** Lowest Test MAE
2. **Secondary:** Lowest Test RMSE
3. **Tertiary:** Highest Test R²

---

## 🔍 **Monitor Jobs**

**Azure ML Studio:** https://ml.azure.com → Jobs

**Command Line:**
```bash
az ml job list --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --max-results 10 \
  --query "[].{Name:name, DisplayName:display_name, Status:status}" \
  --output table
```

---

## ✅ **Next Steps After All Complete**

1. **Compare all models** (check Test MAE)
2. **Pick the best 2-3 models**
3. **Deploy for inference** (Step 3)
4. **A/B test in production**
5. **Iterate and improve**

---

## 🚀 **Ready to Start?**

Run the first two commands to start CatBoost and Random Forest:

```bash
# Start CatBoost NOW
az ml job create --file aml_train_catboost.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

# Start Random Forest NOW (parallel)
az ml job create --file aml_train_random_forest.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

**Both will run in parallel! Total time: ~40 minutes (whichever finishes last)**

---

## 📞 **Need Help?**

Check the logs for any failures:
- Go to https://ml.azure.com
- Find the failed job
- Click "Outputs + logs" → "user_logs" → "std_log.txt"
- Look for the error message

Most common issues:
- ❌ Out of memory → Already using memory-cluster (128GB)
- ❌ Module not found → Need to rebuild Docker image with new dependencies
- ❌ File not found → Check that `ercot_features_manual:1` data asset exists

---

**🎯 Start with CatBoost and Random Forest NOW!** They'll likely beat your current models! 🚀

