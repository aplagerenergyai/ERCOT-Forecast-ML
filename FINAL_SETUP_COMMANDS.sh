#!/bin/bash

# ============================================
# FINAL SETUP - Azure subscription 1 (Owner)
# ============================================

set -e  # Exit on error

echo "🚀 Starting ACR setup in YOUR Owner subscription..."
echo ""

# Step 1: Switch to Owner subscription
echo "Step 1: Switching to Azure subscription 1..."
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f
az account show --query "{Name:name, SubscriptionId:id}" -o table
echo ""

# Step 2: Set variables (CUSTOMIZE IF NEEDED)
RG="rg-ercot-ml-production"  # Change if you want to use existing RG
ACR_NAME="ercotforecastingprod"  # Change if name is taken
LOCATION="southcentralus"

echo "Settings:"
echo "  Resource Group: $RG"
echo "  ACR Name: $ACR_NAME"
echo "  Location: $LOCATION"
echo ""

# Step 3: Create resource group (if doesn't exist)
echo "Step 2: Creating resource group..."
az group create --name $RG --location $LOCATION --output none 2>/dev/null || echo "  Resource group already exists"
echo "  ✅ Resource group ready: $RG"
echo ""

# Step 4: Create ACR
echo "Step 3: Creating Container Registry..."
az acr create \
  --resource-group $RG \
  --name $ACR_NAME \
  --sku Basic \
  --location $LOCATION \
  --admin-enabled true \
  --output none

echo "  ✅ ACR created: $ACR_NAME"
echo ""

# Step 5: Get ML workspace identity
echo "Step 4: Getting ML workspace identity from ennrgyai subscription..."
WORKSPACE_IDENTITY=$(az ml workspace show \
  --name energyaiml \
  --resource-group ennrgyai-rg \
  --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed \
  --query identity.principal_id -o tsv)

if [ -z "$WORKSPACE_IDENTITY" ]; then
    echo "  ❌ ERROR: Could not get workspace identity"
    exit 1
fi

echo "  ✅ Workspace identity: $WORKSPACE_IDENTITY"
echo ""

# Step 6: Grant AcrPull permission
echo "Step 5: Granting AcrPull permission to ML workspace..."
ACR_ID=$(az acr show --name $ACR_NAME --resource-group $RG --query id -o tsv)

az role assignment create \
  --assignee $WORKSPACE_IDENTITY \
  --role AcrPull \
  --scope $ACR_ID \
  --output none

echo "  ✅ Permission granted!"
echo ""

# Step 7: Get ACR credentials
echo "========================================"
echo "✅✅✅ SUCCESS! HERE ARE YOUR VALUES ✅✅✅"
echo "========================================"
echo ""

ACR_USERNAME=$(az acr show --name $ACR_NAME --query name -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

echo "📋 GitHub Secrets to Add:"
echo ""
echo "1. ACR_USERNAME:"
echo "   $ACR_USERNAME"
echo ""
echo "2. ACR_PASSWORD:"
echo "   $ACR_PASSWORD"
echo ""
echo "3. ACR Login Server (already updated in workflow):"
echo "   $ACR_LOGIN_SERVER"
echo ""
echo "4. AZURE_ML_WORKSPACE:"
echo "   energyaiml"
echo ""
echo "5. AZURE_RESOURCE_GROUP:"
echo "   ennrgyai-rg"
echo ""
echo "6. AZURE_SUBSCRIPTION_ID:"
echo "   d06b42ec-a0ae-4415-b534-e6635ac7e7ed"
echo ""

# Step 8: Create service principal
echo "========================================"
echo "Creating Service Principal for GitHub Actions..."
echo "========================================"
echo ""
echo "7. AZURE_CREDENTIALS (copy the ENTIRE JSON below):"
echo ""

az ad sp create-for-rbac \
  --name "github-actions-ercot-ml-prod" \
  --role contributor \
  --scopes /subscriptions/d06b42ec-a0ae-4415-b534-e6635ac7e7ed/resourceGroups/ennrgyai-rg \
  --sdk-auth

echo ""
echo "========================================"
echo "✅ ALL DONE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Copy all 7 values above"
echo "2. Go to GitHub → Settings → Secrets → Actions"
echo "3. Add each secret"
echo "4. git add ."
echo "5. git commit -m \"Setup complete with new ACR\""
echo "6. git push origin main"
echo ""
echo "🎉 Your CI/CD pipeline will work now!"

