#!/bin/bash

################################################################################
# ERCOT ML Training Jobs - Azure ML Submission Script
#
# This script triggers all three training jobs (LightGBM, XGBoost, Deep Learning)
# in Azure ML in parallel.
#
# Usage:
#   ./run_training_jobs.sh
#
# Environment Variables (optional):
#   WORKSPACE_NAME - Azure ML workspace name (default: energyaiml-prod)
#   RESOURCE_GROUP - Resource group name (default: rg-ercot-ml-production)
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_NAME="${WORKSPACE_NAME:-energyaiml-prod}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-ercot-ml-production}"
LOG_DIR=".azureml"
LOG_FILE="${LOG_DIR}/training_jobs.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Header
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ERCOT ML Training Jobs - Azure ML Submission${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Workspace: ${WORKSPACE_NAME}"
echo -e "  Resource Group: ${RESOURCE_GROUP}"
echo -e "  Log File: ${LOG_FILE}"
echo ""

# Initialize log file
echo "================================================" > "${LOG_FILE}"
echo "Training Jobs Submission - $(date)" >> "${LOG_FILE}"
echo "================================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

################################################################################
# Function: Submit Job
################################################################################
submit_job() {
    local job_file="$1"
    local job_name="$2"
    
    echo -e "${YELLOW}Submitting ${job_name}...${NC}"
    echo "----------------------------------------" >> "${LOG_FILE}"
    echo "Job: ${job_name}" >> "${LOG_FILE}"
    echo "File: ${job_file}" >> "${LOG_FILE}"
    echo "Timestamp: $(date)" >> "${LOG_FILE}"
    
    # Submit job and capture output
    output=$(az ml job create \
        --file "${job_file}" \
        --workspace-name "${WORKSPACE_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query "{Name:name, Status:status, DisplayName:display_name}" \
        --output json 2>&1)
    
    if [ $? -eq 0 ]; then
        # Parse job ID from output
        job_id=$(echo "$output" | grep -oP '"Name":\s*"\K[^"]+' || echo "$output" | jq -r '.Name' 2>/dev/null || echo "unknown")
        
        echo -e "${GREEN}✓ Submitted: ${job_id}${NC}"
        echo "Job ID: ${job_id}" >> "${LOG_FILE}"
        echo "Status: Submitted" >> "${LOG_FILE}"
        echo "" >> "${LOG_FILE}"
        
        # Return job ID for tracking
        echo "${job_id}"
    else
        echo -e "${RED}✗ Failed to submit ${job_name}${NC}"
        echo "Error: ${output}" >> "${LOG_FILE}"
        echo "" >> "${LOG_FILE}"
        return 1
    fi
}

################################################################################
# Submit All Jobs
################################################################################

echo -e "${GREEN}Submitting training jobs in parallel...${NC}"
echo ""

# Submit LightGBM
LGBM_JOB=$(submit_job "aml_train_lgbm.yml" "LightGBM Training")
LGBM_STATUS=$?

# Submit XGBoost
XGB_JOB=$(submit_job "aml_train_xgb.yml" "XGBoost Training")
XGB_STATUS=$?

# Submit Deep Learning
DEEP_JOB=$(submit_job "aml_train_deep.yml" "Deep Learning Training")
DEEP_STATUS=$?

################################################################################
# Summary
################################################################################

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Submission Summary${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo "Summary:" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

if [ $LGBM_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ LightGBM:     ${LGBM_JOB}${NC}"
    echo "✓ LightGBM:     ${LGBM_JOB}" >> "${LOG_FILE}"
else
    echo -e "${RED}✗ LightGBM:     Failed${NC}"
    echo "✗ LightGBM:     Failed" >> "${LOG_FILE}"
fi

if [ $XGB_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ XGBoost:      ${XGB_JOB}${NC}"
    echo "✓ XGBoost:      ${XGB_JOB}" >> "${LOG_FILE}"
else
    echo -e "${RED}✗ XGBoost:      Failed${NC}"
    echo "✗ XGBoost:      Failed" >> "${LOG_FILE}"
fi

if [ $DEEP_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Deep Learning: ${DEEP_JOB}${NC}"
    echo "✓ Deep Learning: ${DEEP_JOB}" >> "${LOG_FILE}"
else
    echo -e "${RED}✗ Deep Learning: Failed${NC}"
    echo "✗ Deep Learning: Failed" >> "${LOG_FILE}"
fi

echo ""
echo "================================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

################################################################################
# Monitor Jobs Command
################################################################################

if [ $LGBM_STATUS -eq 0 ] || [ $XGB_STATUS -eq 0 ] || [ $DEEP_STATUS -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}Monitor jobs with:${NC}"
    echo ""
    
    if [ $LGBM_STATUS -eq 0 ]; then
        echo "  az ml job show --name ${LGBM_JOB} --workspace-name ${WORKSPACE_NAME} --resource-group ${RESOURCE_GROUP}"
    fi
    
    if [ $XGB_STATUS -eq 0 ]; then
        echo "  az ml job show --name ${XGB_JOB} --workspace-name ${WORKSPACE_NAME} --resource-group ${RESOURCE_GROUP}"
    fi
    
    if [ $DEEP_STATUS -eq 0 ]; then
        echo "  az ml job show --name ${DEEP_JOB} --workspace-name ${WORKSPACE_NAME} --resource-group ${RESOURCE_GROUP}"
    fi
    
    echo ""
    echo -e "${YELLOW}Or visit Azure ML Studio:${NC}"
    echo "  https://ml.azure.com"
    echo ""
fi

################################################################################
# Exit Status
################################################################################

# Exit with error if any job failed to submit
if [ $LGBM_STATUS -ne 0 ] || [ $XGB_STATUS -ne 0 ] || [ $DEEP_STATUS -ne 0 ]; then
    echo -e "${RED}✗ Some jobs failed to submit. Check ${LOG_FILE} for details.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All training jobs submitted successfully!${NC}"
echo ""
exit 0

