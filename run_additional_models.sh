#!/bin/bash

# ============================================================================
# ERCOT ML - Run 6 Additional Models
# ============================================================================
# This submits 6 additional models beyond the original 5:
# 1. AutoML (tries 13+ algorithms automatically)
# 2. ExtraTrees (CPU)
# 3. HistGradientBoosting (CPU)  
# 4. TabNet (GPU)
# 5. NGBoost (CPU)
# 6. Temporal Fusion Transformer/TFT (GPU)
# ============================================================================

WORKSPACE="energyaiml-prod"
RESOURCE_GROUP="rg-ercot-ml-production"

echo "════════════════════════════════════════════════════════════════"
echo "  🎯 ERCOT ML - Additional 6 Models"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Submitting 6 additional models..."
echo ""

# ============================================================================
# 1. AutoML (HIGHEST PRIORITY - tries 13+ models)
# ============================================================================
echo "1. AutoML - Tries 13+ algorithms automatically"
AUTOML_JOB=$(az ml job create --file aml_train_automl.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${AUTOML_JOB}"
echo "   Expected: ~30 min (memory-cluster)"
echo ""

# ============================================================================
# 2. ExtraTrees (More randomized Random Forest)
# ============================================================================
echo "2. ExtraTrees - More randomized than Random Forest"
EXTRATREES_JOB=$(az ml job create --file aml_train_extratrees.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${EXTRATREES_JOB}"
echo "   Expected: ~40 min (memory-cluster)"
echo ""

# ============================================================================
# 3. HistGradientBoosting (Sklearn's native boosting)
# ============================================================================
echo "3. HistGradientBoosting - Sklearn's fast gradient boosting"
HISTGB_JOB=$(az ml job create --file aml_train_histgb.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${HISTGB_JOB}"
echo "   Expected: ~10 min (memory-cluster)"
echo ""

# ============================================================================
# 4. TabNet (Attention-based deep learning for tabular data)
# ============================================================================
echo "4. TabNet - Attention-based deep learning (GPU)"
TABNET_JOB=$(az ml job create --file aml_train_tabnet.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${TABNET_JOB}"
echo "   Expected: ~60 min (GPU)"
echo ""

# ============================================================================
# 5. NGBoost (Probabilistic gradient boosting with uncertainty)
# ============================================================================
echo "5. NGBoost - Probabilistic gradient boosting"
NGBOOST_JOB=$(az ml job create --file aml_train_ngboost.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${NGBOOST_JOB}"
echo "   Expected: ~20 min (memory-cluster)"
echo ""

# ============================================================================
# 6. Temporal Fusion Transformer (State-of-art time series)
# ============================================================================
echo "6. TFT - Temporal Fusion Transformer (GPU)"
TFT_JOB=$(az ml job create --file aml_train_tft.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${TFT_JOB}"
echo "   Expected: ~120 min (GPU)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ All 6 additional models submitted!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor at: https://ml.azure.com"
echo ""
echo "Job IDs:"
echo "  AutoML:         ${AUTOML_JOB}"
echo "  ExtraTrees:     ${EXTRATREES_JOB}"
echo "  HistGradBoost:  ${HISTGB_JOB}"
echo "  TabNet:         ${TABNET_JOB}"
echo "  NGBoost:        ${NGBOOST_JOB}"
echo "  TFT:            ${TFT_JOB}"
echo ""
echo "Expected completion times:"
echo "  - HistGradBoost: ~10 min"
echo "  - NGBoost: ~20 min"
echo "  - AutoML: ~30 min (PRIORITY)"
echo "  - ExtraTrees: ~40 min"
echo "  - TabNet: ~60 min"
echo "  - TFT: ~120 min"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 PRIORITY: Check AutoML results first (~30 min)"
echo "   AutoML automatically tries 13+ algorithms!"
echo "════════════════════════════════════════════════════════════════"
echo ""

