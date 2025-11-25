#!/bin/bash

# ERCOT ML Pipeline - Quick Status Check
# Run this when you come back in 4-5 hours

echo "════════════════════════════════════════════════════════════════"
echo "  ERCOT ML Pipeline - Status Check"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
FEATURE_JOB="willing_hominy_5mkqggcb33"
WORKSPACE="energyaiml-prod"
RESOURCE_GROUP="rg-ercot-ml-production"

# Check feature engineering job status
echo "🔍 Checking Feature Engineering Job Status..."
STATUS=$(az ml job show --name $FEATURE_JOB --workspace-name $WORKSPACE --resource-group $RESOURCE_GROUP --query status -o tsv 2>/dev/null)

if [ -z "$STATUS" ]; then
    echo "❌ Could not get job status. Check your Azure CLI login."
    exit 1
fi

echo "   Job ID: $FEATURE_JOB"
echo "   Status: $STATUS"
echo ""

if [ "$STATUS" == "Completed" ]; then
    echo "✅ ✅ ✅  FEATURE ENGINEERING COMPLETED!  ✅ ✅ ✅"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Next Steps:"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "1️⃣  Download and Verify Output:"
    echo ""
    echo "    az ml job download --name $FEATURE_JOB \\"
    echo "      --workspace-name $WORKSPACE \\"
    echo "      --resource-group $RESOURCE_GROUP \\"
    echo "      --download-path ./features_output \\"
    echo "      --output-name features"
    echo ""
    echo "2️⃣  Check File Size:"
    echo ""
    echo "    if [ -f \"./features_output/named-outputs/features/hourly_features.parquet\" ]; then"
    echo "        ls -lh ./features_output/named-outputs/features/hourly_features.parquet | awk '{print \"Size: \" \$5}'"
    echo "    else"
    echo "        echo \"❌ File not found\""
    echo "    fi"
    echo ""
    echo "3️⃣  Validate Data Quality:"
    echo ""
    echo "    python validate_parquet.py --file ./features_output/named-outputs/features/hourly_features.parquet"
    echo ""
    echo "4️⃣  Get the new feature path to update training YAMLs:"
    echo ""
    echo "    # Look in Azure ML Studio job outputs for the UUID"
    echo "    # Or check the std_log.txt for the output path"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    
elif [ "$STATUS" == "Running" ]; then
    echo "🏃 Still Running..."
    echo ""
    echo "   Check back later, or monitor in Azure ML Studio:"
    echo "   https://ml.azure.com"
    echo ""
    
elif [ "$STATUS" == "Failed" ]; then
    echo "❌ FAILED!"
    echo ""
    echo "   Check the logs in Azure ML Studio:"
    echo "   https://ml.azure.com → Jobs → $FEATURE_JOB"
    echo "   Look at: user_logs/std_log.txt"
    echo ""
    
else
    echo "⏳ Status: $STATUS"
    echo ""
    echo "   Current status is: $STATUS"
    echo "   Check back later."
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"

