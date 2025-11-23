#!/bin/bash

echo "========================================"
echo "Checking ALL Subscriptions for Owner Rights"
echo "========================================"
echo ""

MY_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv)
echo "Your Object ID: $MY_OBJECT_ID"
echo ""

# Get all subscriptions
SUBSCRIPTIONS=$(az account list --query "[].{Name:name, SubscriptionId:id}" -o tsv)

echo "$SUBSCRIPTIONS" | while IFS=$'\t' read -r NAME SUB_ID; do
    echo "========================================"
    echo "Subscription: $NAME"
    echo "ID: $SUB_ID"
    echo "========================================"
    
    # Switch to this subscription
    az account set --subscription "$SUB_ID"
    
    # Check your roles
    echo "Your Roles:"
    az role assignment list \
        --assignee $MY_OBJECT_ID \
        --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
    
    # Check for Owner or UAA
    OWNER_CHECK=$(az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='Owner']" -o tsv)
    UAA_CHECK=$(az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='User Access Administrator']" -o tsv)
    
    echo ""
    if [ -n "$OWNER_CHECK" ]; then
        echo "✅✅✅ YOU HAVE OWNER ROLE IN THIS SUBSCRIPTION ✅✅✅" 
        echo "USE THIS ONE!"
    elif [ -n "$UAA_CHECK" ]; then
        echo "✅ You have User Access Administrator"
    else
        echo "❌ No elevated permissions in this subscription"
    fi
    
    echo ""
    echo ""
done

# Switch back to default
az account set --subscription "d06b42ec-a0ae-4415-b534-e6635ac7e7ed"

