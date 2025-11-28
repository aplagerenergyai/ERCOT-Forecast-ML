#!/bin/bash

# ============================================================================
# ERCOT ML - Run ALL Models on GPU (One Command)
# ============================================================================
# This submits all 9 models:
# - 7 GPU models in parallel (CatBoost, Deep, LightGBM, XGBoost, TwoStage, RegimeClassifier, Quantile)
# - 1 CPU model in parallel (Random Forest)
# - 1 Ensemble after others complete
# Total time: ~45 minutes
# ============================================================================

WORKSPACE="energyaiml-prod"
RESOURCE_GROUP="rg-ercot-ml-production"

echo "════════════════════════════════════════════════════════════════"
echo "  🎮 ERCOT ML - GPU Training (All Models)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Submitting 8 models in parallel..."
echo ""

# Submit all models that can run in parallel
echo "1. CatBoost (GPU) - ~8 min"
CATBOOST_JOB=$(az ml job create --file aml_train_catboost.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${CATBOOST_JOB}"

echo "2. Deep Learning (GPU) - ~10 min"
DEEP_JOB=$(az ml job create --file aml_train_deep.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${DEEP_JOB}"

echo "3. LightGBM (GPU) - ~1 min"
LGBM_JOB=$(az ml job create --file aml_train_lgbm.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${LGBM_JOB}"

echo "4. XGBoost (GPU) - ~1 min"
XGB_JOB=$(az ml job create --file aml_train_xgb.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${XGB_JOB}"

echo "5. Random Forest (CPU) - ~40 min"
RF_JOB=$(az ml job create --file aml_train_random_forest.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${RF_JOB}"

echo "6. Two-Stage Model (GPU) - ~2 min"
TWOSTAGE_JOB=$(az ml job create --file aml_train_two_stage.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${TWOSTAGE_JOB}"

echo "7. Regime Classifier (GPU) - ~2 min"
REGIME_JOB=$(az ml job create --file aml_train_regime_classifier.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${REGIME_JOB}"

echo "8. Quantile Regression (GPU) - ~3 min"
QUANTILE_JOB=$(az ml job create --file aml_train_quantile.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "   ✓ Job ID: ${QUANTILE_JOB}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ All 8 models submitted!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor at: https://ml.azure.com"
echo ""
echo "Job IDs:"
echo "  CatBoost:          ${CATBOOST_JOB}"
echo "  Deep Learning:     ${DEEP_JOB}"
echo "  LightGBM:          ${LGBM_JOB}"
echo "  XGBoost:           ${XGB_JOB}"
echo "  Random Forest:     ${RF_JOB}"
echo "  Two-Stage:         ${TWOSTAGE_JOB}"
echo "  Regime Classifier: ${REGIME_JOB}"
echo "  Quantile:          ${QUANTILE_JOB}"
echo ""
echo "Expected completion: ~45 minutes (Random Forest takes longest)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ⏳ Waiting for all jobs to complete..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Function to check if a job is complete
check_job_status() {
    local job_id=$1
    az ml job show --name ${job_id} \
      --workspace-name ${WORKSPACE} \
      --resource-group ${RESOURCE_GROUP} \
      --query status -o tsv 2>/dev/null
}

# Wait for all jobs to complete
while true; do
    CATBOOST_STATUS=$(check_job_status ${CATBOOST_JOB})
    DEEP_STATUS=$(check_job_status ${DEEP_JOB})
    LGBM_STATUS=$(check_job_status ${LGBM_JOB})
    XGB_STATUS=$(check_job_status ${XGB_JOB})
    RF_STATUS=$(check_job_status ${RF_JOB})
    TWOSTAGE_STATUS=$(check_job_status ${TWOSTAGE_JOB})
    REGIME_STATUS=$(check_job_status ${REGIME_JOB})
    QUANTILE_STATUS=$(check_job_status ${QUANTILE_JOB})
    
    echo "Status:"
    echo "  CatBoost:          ${CATBOOST_STATUS}"
    echo "  Deep Learning:     ${DEEP_STATUS}"
    echo "  LightGBM:          ${LGBM_STATUS}"
    echo "  XGBoost:           ${XGB_STATUS}"
    echo "  Random Forest:     ${RF_STATUS}"
    echo "  Two-Stage:         ${TWOSTAGE_STATUS}"
    echo "  Regime Classifier: ${REGIME_STATUS}"
    echo "  Quantile:          ${QUANTILE_STATUS}"
    echo ""
    
    # Check if all are complete
    if [[ "${CATBOOST_STATUS}" == "Completed" ]] && \
       [[ "${DEEP_STATUS}" == "Completed" ]] && \
       [[ "${LGBM_STATUS}" == "Completed" ]] && \
       [[ "${XGB_STATUS}" == "Completed" ]] && \
       [[ "${RF_STATUS}" == "Completed" ]] && \
       [[ "${TWOSTAGE_STATUS}" == "Completed" ]] && \
       [[ "${REGIME_STATUS}" == "Completed" ]] && \
       [[ "${QUANTILE_STATUS}" == "Completed" ]]; then
        echo "✅ All models completed!"
        break
    fi
    
    # Check for failures
    if [[ "${CATBOOST_STATUS}" == "Failed" ]] || \
       [[ "${DEEP_STATUS}" == "Failed" ]] || \
       [[ "${LGBM_STATUS}" == "Failed" ]] || \
       [[ "${XGB_STATUS}" == "Failed" ]] || \
       [[ "${RF_STATUS}" == "Failed" ]] || \
       [[ "${TWOSTAGE_STATUS}" == "Failed" ]] || \
       [[ "${REGIME_STATUS}" == "Failed" ]] || \
       [[ "${QUANTILE_STATUS}" == "Failed" ]]; then
        echo "❌ One or more jobs failed. Check logs at https://ml.azure.com"
        exit 1
    fi
    
    echo "Checking again in 60 seconds..."
    sleep 60
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎯 Submitting Ensemble Model"
echo "════════════════════════════════════════════════════════════════"
echo ""

ENSEMBLE_JOB=$(az ml job create --file aml_train_ensemble.yml \
  --workspace-name ${WORKSPACE} \
  --resource-group ${RESOURCE_GROUP} \
  --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
echo "✓ Ensemble Job ID: ${ENSEMBLE_JOB}"

echo ""
echo "Waiting for ensemble to complete..."
while true; do
    ENSEMBLE_STATUS=$(check_job_status ${ENSEMBLE_JOB})
    echo "  Status: ${ENSEMBLE_STATUS}"
    
    if [[ "${ENSEMBLE_STATUS}" == "Completed" ]]; then
        echo "✅ Ensemble completed!"
        break
    elif [[ "${ENSEMBLE_STATUS}" == "Failed" ]]; then
        echo "❌ Ensemble failed. Check logs at https://ml.azure.com"
        exit 1
    fi
    
    sleep 30
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎉 ALL MODELS COMPLETED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Check results at: https://ml.azure.com"
echo ""
echo "Compare Test Set Metrics (MAE) to find the best model:"
echo "  - Lower MAE = Better"
echo "  - Current best: LightGBM at \$11.90 MAE"
echo "  - Target: < \$11.00 MAE"
echo ""

