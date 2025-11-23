#!/bin/bash

echo "================================"
echo "Azure Permissions Diagnostic"
echo "================================"
echo ""

# 1. Check current account
echo "1. Current Azure Account:"
az account show --query "{Name:name, User:user.name, SubscriptionId:id, State:state}" -o table
echo ""

# 2. Check current subscription
echo "2. Current Subscription:"
CURRENT_SUB=$(az account show --query id -o tsv)
echo "Subscription ID: $CURRENT_SUB"
echo ""

# 3. List all subscriptions
echo "3. All Available Subscriptions:"
az account list --query "[].{Name:name, SubscriptionId:id, IsDefault:isDefault}" -o table
echo ""

# 4. Check your role assignments
echo "4. Your Role Assignments in Current Subscription:"
MY_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv)
az role assignment list --assignee $MY_OBJECT_ID --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
echo ""

# 5. Check if you have Owner or User Access Administrator
echo "5. Checking for elevated roles:"
OWNER_CHECK=$(az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='Owner']" -o tsv)
UAA_CHECK=$(az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='User Access Administrator']" -o tsv)

if [ -n "$OWNER_CHECK" ]; then
    echo "✅ You have OWNER role"
else
    echo "❌ You do NOT have OWNER role"
fi

if [ -n "$UAA_CHECK" ]; then
    echo "✅ You have USER ACCESS ADMINISTRATOR role"
else
    echo "❌ You do NOT have USER ACCESS ADMINISTRATOR role"
fi
echo ""

# 6. Check specific resource group
echo "6. Your Roles in Resource Group 'ennrgyai-rg':"
az role assignment list \
  --assignee $MY_OBJECT_ID \
  --resource-group ennrgyai-rg \
  --query "[].{Role:roleDefinitionName, Scope:scope}" -o table 2>/dev/null || echo "⚠️  Cannot access resource group or it doesn't exist"
echo ""

# 7. Check if ACR exists and your access
echo "7. Checking ACR 'ercotforecasting':"
az acr show --name ercotforecasting --query "{Name:name, ResourceGroup:resourceGroup, LoginServer:loginServer}" -o table 2>/dev/null || echo "⚠️  Cannot access ACR or it doesn't exist"
echo ""

# 8. Try to get ACR credentials
echo "8. Testing ACR Credential Access:"
az acr credential show --name ercotforecasting --query "{Username:username}" -o table 2>/dev/null && echo "✅ You can access ACR credentials" || echo "❌ Cannot access ACR credentials"
echo ""

# 9. Check ML workspace
echo "9. Checking ML Workspace 'energyaiml':"
az ml workspace show --name energyaiml --resource-group ennrgyai-rg --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" -o table 2>/dev/null || echo "⚠️  Cannot access ML workspace"
echo ""

echo "================================"
echo "Diagnostic Complete"
echo "================================"

