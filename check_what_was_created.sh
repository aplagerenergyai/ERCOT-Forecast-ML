#!/bin/bash

echo "========================================"
echo "Checking What Was Created..."
echo "========================================"
echo ""

# Switch to Owner subscription
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f

echo "Current subscription:"
az account show --query "{Name:name, SubscriptionId:id}" -o table
echo ""

# Check for ACR
echo "========================================"
echo "1. Checking for ACR 'ercotforecastingprod':"
echo "========================================"

ACR_EXISTS=$(az acr show --name ercotforecastingprod --query name -o tsv 2>/dev/null)

if [ -z "$ACR_EXISTS" ]; then
    echo "❌ ACR 'ercotforecastingprod' NOT FOUND"
    echo ""
    echo "Checking for ANY ACRs in this subscription:"
    az acr list --query "[].{Name:name, ResourceGroup:resourceGroup, LoginServer:loginServer}" -o table
else
    echo "✅ ACR EXISTS!"
    az acr show --name ercotforecastingprod --query "{Name:name, ResourceGroup:resourceGroup, LoginServer:loginServer, Location:location}" -o table
    
    # Get credentials
    echo ""
    echo "ACR Username:"
    echo "ercotforecastingprod"
    
    echo ""
    echo "ACR Password:"
    az acr credential show --name ercotforecastingprod --query "passwords[0].value" -o tsv
    
    echo ""
    echo "ACR Login Server:"
    az acr show --name ercotforecastingprod --query loginServer -o tsv
fi

echo ""
echo ""

# Check for resource group
echo "========================================"
echo "2. Checking for Resource Group 'rg-ercot-ml-production':"
echo "========================================"

RG_EXISTS=$(az group show --name rg-ercot-ml-production --query name -o tsv 2>/dev/null)

if [ -z "$RG_EXISTS" ]; then
    echo "❌ Resource group 'rg-ercot-ml-production' NOT FOUND"
    echo ""
    echo "All resource groups in this subscription:"
    az group list --query "[].{Name:name, Location:location}" -o table
else
    echo "✅ Resource group EXISTS!"
    az group show --name rg-ercot-ml-production --query "{Name:name, Location:location}" -o table
fi

echo ""
echo ""

# Check for AcrPull permission
echo "========================================"
echo "3. Checking AcrPull Permission:"
echo "========================================"

if [ ! -z "$ACR_EXISTS" ]; then
    ACR_ID=$(az acr show --name ercotforecastingprod --query id -o tsv)
    
    ACRPULL_EXISTS=$(az role assignment list \
        --scope $ACR_ID \
        --query "[?roleDefinitionName=='AcrPull']" -o tsv)
    
    if [ -z "$ACRPULL_EXISTS" ]; then
        echo "❌ No AcrPull permissions found"
    else
        echo "✅ AcrPull permission EXISTS!"
        az role assignment list \
            --scope $ACR_ID \
            --query "[?roleDefinitionName=='AcrPull'].{Role:roleDefinitionName, Principal:principalName}" -o table
    fi
else
    echo "⚠️  Cannot check (ACR doesn't exist)"
fi

echo ""
echo ""

# Check for service principal
echo "========================================"
echo "4. Checking for Service Principal:"
echo "========================================"

SP_EXISTS=$(az ad sp list --display-name "github-actions-ercot-ml-prod" --query "[0].appId" -o tsv 2>/dev/null)

if [ -z "$SP_EXISTS" ]; then
    echo "❌ Service principal 'github-actions-ercot-ml-prod' NOT FOUND"
else
    echo "✅ Service principal EXISTS!"
    echo "App ID: $SP_EXISTS"
fi

echo ""
echo ""

# Summary
echo "========================================"
echo "SUMMARY:"
echo "========================================"
echo ""

NEEDS_ACR=false
NEEDS_PERMISSION=false
NEEDS_SP=false

if [ -z "$ACR_EXISTS" ]; then
    echo "❌ Need to create ACR"
    NEEDS_ACR=true
else
    echo "✅ ACR created"
fi

if [ -z "$ACRPULL_EXISTS" ]; then
    echo "❌ Need to grant AcrPull permission"
    NEEDS_PERMISSION=true
else
    echo "✅ AcrPull permission granted"
fi

if [ -z "$SP_EXISTS" ]; then
    echo "❌ Need to create service principal"
    NEEDS_SP=true
else
    echo "✅ Service principal created"
fi

echo ""

if [ "$NEEDS_ACR" = false ] && [ "$NEEDS_PERMISSION" = false ] && [ "$NEEDS_SP" = false ]; then
    echo "🎉🎉🎉 EVERYTHING IS DONE! 🎉🎉🎉"
    echo ""
    echo "Just add the GitHub secrets (values shown above) and push!"
else
    echo "⚠️  Some steps still need to be completed."
    echo ""
    echo "Run these commands to finish:"
    echo ""
    
    if [ "$NEEDS_ACR" = true ]; then
        echo "# Create ACR:"
        echo "az acr create --resource-group rg-ercot-ml-production --name ercotforecastingprod --sku Basic --location southcentralus --admin-enabled true"
        echo ""
    fi
    
    if [ "$NEEDS_PERMISSION" = true ]; then
        echo "# Grant AcrPull permission:"
        echo "WORKSPACE_IDENTITY=\$(az ml workspace show --name energyaiml --resource-group ennrgyai-rg --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed --query identity.principal_id -o tsv)"
        echo "az role assignment create --assignee \$WORKSPACE_IDENTITY --role AcrPull --scope \$(az acr show --name ercotforecastingprod --query id -o tsv)"
        echo ""
    fi
    
    if [ "$NEEDS_SP" = true ]; then
        echo "# Create service principal:"
        echo "az ad sp create-for-rbac --name \"github-actions-ercot-ml-prod\" --role contributor --scopes /subscriptions/d06b42ec-a0ae-4415-b534-e6635ac7e7ed/resourceGroups/ennrgyai-rg --sdk-auth"
        echo ""
    fi
fi

