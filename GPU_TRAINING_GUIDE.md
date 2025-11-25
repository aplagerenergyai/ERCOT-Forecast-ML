# 🎮 GPU Training Guide - ERCOT ML Pipeline

## ✅ **GPU Cluster Configured!**

**Your GPU Cluster:** `GPUClusterNC8asT4v3`
- **GPU:** NVIDIA Tesla T4 (16GB vRAM)
- **CPU:** 8 cores, 56GB RAM
- **Cost:** ~$0.90/hour (much cheaper than V100!)

---

## 🚀 **GPU Benefits:**

| Model | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| **LightGBM** | ~3 min | ~1 min | **3x faster** |
| **XGBoost** | ~3 min | ~1 min | **3x faster** |
| **CatBoost** | ~30 min | ~8 min | **4x faster** |
| **Deep Learning** | ~60 min | ~10 min | **6x faster** |
| **Random Forest** | ~40 min | ~40 min | No GPU support |

**Total Savings:** 2+ hours → 1 hour (50% faster!)

---

## ✅ **What I've Updated:**

### **1. Training Scripts (GPU-Enabled):**
- ✅ `train_lgbm.py` - Added `device: 'gpu'`
- ✅ `train_xgb.py` - Added `tree_method: 'gpu_hist'`, `device: 'cuda'`
- ✅ `train_catboost.py` - Changed to `task_type: 'GPU'`
- ✅ `train_deep.py` - Already had CUDA detection
- ⚠️ `train_random_forest.py` - No GPU support (uses CPU)

### **2. Azure ML YAML Files:**
- ✅ `aml_train_lgbm.yml` → `compute: azureml:GPUClusterNC8asT4v3`
- ✅ `aml_train_xgb.yml` → `compute: azureml:GPUClusterNC8asT4v3`
- ✅ `aml_train_catboost.yml` → `compute: azureml:GPUClusterNC8asT4v3`
- ✅ `aml_train_deep.yml` → `compute: azureml:GPUClusterNC8asT4v3`
- ✅ `aml_train_ensemble.yml` → `compute: azureml:GPUClusterNC8asT4v3`
- ⚠️ `aml_train_random_forest.yml` → Still uses `memory-cluster` (no GPU benefit)

---

## 🚀 **Quick Start Commands**

### **Step 1: Rebuild Docker Image (Required!)**

The training scripts were updated to use GPU, so rebuild:

```bash
# Option A: Push to GitHub (auto-builds)
git add .
git commit -m "Enable GPU acceleration for all models"
git push
```

**OR**

```bash
# Option B: Manual build (faster)
docker build -t ercotforecastingprod.azurecr.io/ercot-ml-pipeline:latest .
docker push ercotforecastingprod.azurecr.io/ercot-ml-pipeline:latest
```

---

### **Step 2: Run All Models on GPU**

#### **🎯 Priority 1: CatBoost (8 min with GPU)**
```bash
az ml job create --file aml_train_catboost.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

#### **🎯 Priority 2: Deep Learning (10 min with GPU)**
```bash
# Cancel the current CPU job if still running:
az ml job cancel --name sincere_roof_zwl7x9qwk2 \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production

# Submit GPU version:
az ml job create --file aml_train_deep.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

#### **🎯 Priority 3: LightGBM & XGBoost (1 min each with GPU)**
```bash
# Rerun on GPU for comparison
az ml job create --file aml_train_lgbm.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv

az ml job create --file aml_train_xgb.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

#### **🎯 Priority 4: Random Forest (40 min on CPU)**
```bash
# Stays on memory-cluster (no GPU benefit)
az ml job create --file aml_train_random_forest.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

#### **🎯 Priority 5: Ensemble (5 min)**
```bash
# After all models complete
az ml job create --file aml_train_ensemble.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

---

## 🎯 **Recommended Execution Plan**

### **Phase 1: GPU Models (Parallel - ~10 minutes total)**
Submit all GPU models at once (they'll run on separate GPU nodes):

```bash
# Submit all GPU models in parallel
az ml job create --file aml_train_catboost.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv

az ml job create --file aml_train_deep.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv

az ml job create --file aml_train_lgbm.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv

az ml job create --file aml_train_xgb.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
```

**Wait ~10 minutes for all to complete.**

---

### **Phase 2: CPU Model (Parallel - ~40 minutes)**
While GPU models run, also start Random Forest on the CPU cluster:

```bash
az ml job create --file aml_train_random_forest.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

**This runs on `memory-cluster` in parallel with GPU jobs.**

---

### **Phase 3: Ensemble (~5 minutes)**
After all models complete:

```bash
az ml job create --file aml_train_ensemble.yml \
  --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --query name -o tsv
```

---

## ⏱️ **Total Time:**

**Without GPU:** 2+ hours sequentially
**With GPU:** **~45 minutes** (40 min for Random Forest, others finish in 10 min)

---

## 📊 **How to Verify GPU Usage**

After jobs start, check the logs for GPU confirmation:

### **CatBoost:**
```
task_type: GPU
devices: 0
```

### **XGBoost:**
```
tree_method: gpu_hist
device: cuda
```

### **LightGBM:**
```
device: gpu
gpu_platform_id: 0
```

### **Deep Learning:**
```
Using device: cuda
```

---

## 🔍 **Monitor Jobs**

```bash
# Check all running jobs
az ml job list --workspace-name energyaiml-prod \
  --resource-group rg-ercot-ml-production \
  --max-results 10 \
  --query "[].{Name:name, DisplayName:display_name, Status:status}" \
  --output table
```

**Azure ML Studio:** https://ml.azure.com → Jobs

---

## 💰 **Cost Savings**

**Before (CPU only):**
- memory-cluster: $1.44/hour × 2 hours = **$2.88**

**After (GPU):**
- GPUClusterNC8asT4v3: $0.90/hour × 0.75 hours = **$0.68** (GPU jobs)
- memory-cluster: $1.44/hour × 0.67 hours = **$0.96** (Random Forest)
- **Total: $1.64** (43% cheaper + 60% faster!)

---

## ❓ **Troubleshooting**

### **Error: GPU not available**
Check logs for:
```
CUDA not available, falling back to CPU
```

**Fix:** Ensure Docker image has CUDA support (PyTorch with CUDA)

---

### **Error: Out of GPU memory**
T4 has 16GB vRAM, should be enough. If you see OOM:
- Reduce batch size in Deep Learning
- Reduce tree depth in XGBoost/CatBoost

---

## ✅ **Ready to Start?**

1. **Rebuild Docker image** (5 min)
2. **Submit all GPU models** (parallel - 10 min)
3. **Submit Random Forest** (parallel - 40 min)
4. **Submit Ensemble** (5 min after others)

**Total: ~45 minutes** 🎉

---

## 🚀 **COPY-PASTE COMMANDS:**

```bash
# 1. Rebuild Docker (choose one):
git add . && git commit -m "Enable GPU" && git push
# OR
docker build -t ercotforecastingprod.azurecr.io/ercot-ml-pipeline:latest . && docker push ercotforecastingprod.azurecr.io/ercot-ml-pipeline:latest

# 2. Submit all models (wait for Docker rebuild first!)
az ml job create --file aml_train_catboost.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
az ml job create --file aml_train_deep.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
az ml job create --file aml_train_lgbm.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
az ml job create --file aml_train_xgb.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
az ml job create --file aml_train_random_forest.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv

# 3. Wait ~45 minutes, then submit ensemble:
az ml job create --file aml_train_ensemble.yml --workspace-name energyaiml-prod --resource-group rg-ercot-ml-production --query name -o tsv
```

---

**🎮 You're all set to train on GPU! 🚀**

