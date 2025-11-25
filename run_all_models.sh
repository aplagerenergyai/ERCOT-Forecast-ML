#!/bin/bash

# ============================================================================
# ERCOT ML - Run All Model Training Jobs
# ============================================================================
# This script submits all model training jobs to Azure ML in sequence:
# 1. CatBoost (Priority 1 - Best categorical handling)
# 2. Random Forest (Priority 2 - Robust to outliers)
# 3. Ensemble (Priority 3 - Combines all models)
# ============================================================================

# Configuration
WORKSPACE="energyaiml-prod"
RESOURCE_GROUP="rg-ercot-ml-production"
LOG_FILE=".azureml/all_training_jobs.log"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure .azureml directory exists
mkdir -p .azureml

echo "════════════════════════════════════════════════════════════════"
echo "  ERCOT ML - Running ALL Model Training Jobs"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Workspace: ${WORKSPACE}"
echo "  Resource Group: ${RESOURCE_GROUP}"
echo "  Log File: ${LOG_FILE}"
echo ""

# Clear previous logs
> "${LOG_FILE}"

# Function to submit a job and wait for completion
submit_and_wait() {
    local model_name=$1
    local yaml_file=$2
    local wait_time=$3
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${BLUE}[$model_name] Submitting...${NC}"
    echo "════════════════════════════════════════════════════════════════"
    
    JOB_ID=$(az ml job create --file "${yaml_file}" \
        --workspace-name "${WORKSPACE}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query name -o tsv 2>&1 | grep -v "^Class " | tail -1)
    
    if [ -n "${JOB_ID}" ] && [ "${JOB_ID}" != "" ]; then
        echo -e "${GREEN}✓ Submitted: ${JOB_ID}${NC}"
        echo "${model_name}: ${JOB_ID}" >> "${LOG_FILE}"
        echo "  Monitor at: https://ml.azure.com"
        echo "  Expected time: ${wait_time} minutes"
        echo ""
        echo "  Waiting for completion..."
        
        # Poll job status
        while true; do
            STATUS=$(az ml job show --name "${JOB_ID}" \
                --workspace-name "${WORKSPACE}" \
                --resource-group "${RESOURCE_GROUP}" \
                --query status -o tsv 2>/dev/null)
            
            if [ "${STATUS}" == "Completed" ]; then
                echo -e "${GREEN}✓ ${model_name} COMPLETED${NC}"
                break
            elif [ "${STATUS}" == "Failed" ]; then
                echo -e "❌ ${model_name} FAILED"
                echo "  Check logs at: https://ml.azure.com"
                break
            elif [ "${STATUS}" == "Canceled" ]; then
                echo -e "⚠️  ${model_name} CANCELED"
                break
            fi
            
            echo "  Status: ${STATUS} (checking again in 60s...)"
            sleep 60
        done
    else
        echo -e "❌ Failed to submit ${model_name}"
        echo "  Error: ${JOB_ID}"
    fi
}

# ============================================================================
# Submit Models in Priority Order
# ============================================================================

echo "Starting model training sequence..."
echo ""

# Model 1: CatBoost (Priority 1)
submit_and_wait "CatBoost" "aml_train_catboost.yml" "25-35"

# Model 2: Random Forest (Priority 2)
submit_and_wait "Random Forest" "aml_train_random_forest.yml" "30-45"

# Model 3: Ensemble (Priority 3 - requires other models)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Note: Ensemble requires LightGBM, XGBoost, CatBoost, and RF"
echo "  Make sure those models completed successfully before running"
echo "════════════════════════════════════════════════════════════════"
echo ""
read -p "Press Enter to submit Ensemble job (or Ctrl+C to skip)..."
submit_and_wait "Ensemble" "aml_train_ensemble.yml" "5-10"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Training Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
cat "${LOG_FILE}"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All jobs submitted!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "View all jobs at: https://ml.azure.com"
echo ""
echo "To compare models, check the Test Set Metrics in each job's logs:"
echo "  - Test RMSE (lower is better)"
echo "  - Test MAE (lower is better)"  
echo "  - Test R² (higher is better)"
echo ""

