# Azure Permissions Diagnostic Script
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Azure Permissions Diagnostic" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check current account
Write-Host "1. Current Azure Account:" -ForegroundColor Yellow
az account show --query "{Name:name, User:user.name, SubscriptionId:id, State:state}" -o table
Write-Host ""

# 2. Check current subscription
Write-Host "2. Current Subscription:" -ForegroundColor Yellow
$CURRENT_SUB = az account show --query id -o tsv
Write-Host "Subscription ID: $CURRENT_SUB"
Write-Host ""

# 3. List all subscriptions
Write-Host "3. All Available Subscriptions:" -ForegroundColor Yellow
az account list --query "[].{Name:name, SubscriptionId:id, IsDefault:isDefault}" -o table
Write-Host ""

# 4. Check your role assignments
Write-Host "4. Your Role Assignments in Current Subscription:" -ForegroundColor Yellow
$MY_OBJECT_ID = az ad signed-in-user show --query id -o tsv
az role assignment list --assignee $MY_OBJECT_ID --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
Write-Host ""

# 5. Check if you have Owner or User Access Administrator
Write-Host "5. Checking for elevated roles:" -ForegroundColor Yellow
$OWNER_CHECK = az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='Owner']" -o tsv
$UAA_CHECK = az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='User Access Administrator']" -o tsv

if ($OWNER_CHECK) {
    Write-Host "✅ You have OWNER role" -ForegroundColor Green
} else {
    Write-Host "❌ You do NOT have OWNER role" -ForegroundColor Red
}

if ($UAA_CHECK) {
    Write-Host "✅ You have USER ACCESS ADMINISTRATOR role" -ForegroundColor Green
} else {
    Write-Host "❌ You do NOT have USER ACCESS ADMINISTRATOR role" -ForegroundColor Red
}
Write-Host ""

# 6. Check specific resource group
Write-Host "6. Your Roles in Resource Group 'ennrgyai-rg':" -ForegroundColor Yellow
try {
    az role assignment list `
      --assignee $MY_OBJECT_ID `
      --resource-group ennrgyai-rg `
      --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
} catch {
    Write-Host "⚠️  Cannot access resource group or it doesn't exist" -ForegroundColor Yellow
}
Write-Host ""

# 7. Check if ACR exists and your access
Write-Host "7. Checking ACR 'ercotforecasting':" -ForegroundColor Yellow
try {
    az acr show --name ercotforecasting --query "{Name:name, ResourceGroup:resourceGroup, LoginServer:loginServer}" -o table
} catch {
    Write-Host "⚠️  Cannot access ACR or it doesn't exist" -ForegroundColor Yellow
}
Write-Host ""

# 8. Try to get ACR credentials
Write-Host "8. Testing ACR Credential Access:" -ForegroundColor Yellow
try {
    az acr credential show --name ercotforecasting --query "{Username:username}" -o table
    Write-Host "✅ You can access ACR credentials" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot access ACR credentials" -ForegroundColor Red
}
Write-Host ""

# 9. Check ML workspace
Write-Host "9. Checking ML Workspace 'energyaiml':" -ForegroundColor Yellow
try {
    az ml workspace show --name energyaiml --resource-group ennrgyai-rg --query "{Name:name, ResourceGroup:resourceGroup, Location:location}" -o table
} catch {
    Write-Host "⚠️  Cannot access ML workspace" -ForegroundColor Yellow
}
Write-Host ""

# 10. Check if you're in the right subscription
Write-Host "10. Subscription Check:" -ForegroundColor Yellow
$TARGET_SUB = "d06b42ec-a0ae-4415-b534-e6635ac7e7ed"
if ($CURRENT_SUB -eq $TARGET_SUB) {
    Write-Host "✅ You are in the correct subscription" -ForegroundColor Green
} else {
    Write-Host "❌ WRONG SUBSCRIPTION! You need to switch to: $TARGET_SUB" -ForegroundColor Red
    Write-Host "Run: az account set --subscription $TARGET_SUB" -ForegroundColor Cyan
}
Write-Host ""

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Diagnostic Complete" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "QUICK FIXES:" -ForegroundColor Yellow
Write-Host "1. Wrong subscription? Run: az account set --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed" -ForegroundColor White
Write-Host "2. Stale credentials? Run: az logout, then az login" -ForegroundColor White
Write-Host "3. Need to refresh token? Run: az account get-access-token" -ForegroundColor White

