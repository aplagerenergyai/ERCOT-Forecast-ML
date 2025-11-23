# Check ALL Subscriptions for Owner Rights
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Checking ALL Subscriptions for Owner Rights" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$MY_OBJECT_ID = az ad signed-in-user show --query id -o tsv
Write-Host "Your Object ID: $MY_OBJECT_ID" -ForegroundColor Yellow
Write-Host ""

# Get all subscriptions
$SUBSCRIPTIONS = az account list --query "[].{Name:name, SubscriptionId:id}" | ConvertFrom-Json

foreach ($SUB in $SUBSCRIPTIONS) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Subscription: $($SUB.Name)" -ForegroundColor White
    Write-Host "ID: $($SUB.SubscriptionId)" -ForegroundColor White
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Switch to this subscription
    az account set --subscription $SUB.SubscriptionId
    
    # Check your roles
    Write-Host "Your Roles:" -ForegroundColor Yellow
    az role assignment list `
        --assignee $MY_OBJECT_ID `
        --query "[].{Role:roleDefinitionName, Scope:scope}" -o table
    
    # Check for Owner or UAA
    $OWNER_CHECK = az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='Owner']" -o tsv
    $UAA_CHECK = az role assignment list --assignee $MY_OBJECT_ID --query "[?roleDefinitionName=='User Access Administrator']" -o tsv
    
    Write-Host ""
    if ($OWNER_CHECK) {
        Write-Host "✅✅✅ YOU HAVE OWNER ROLE IN THIS SUBSCRIPTION ✅✅✅" -ForegroundColor Green -BackgroundColor Black
        Write-Host "USE THIS ONE!" -ForegroundColor Green -BackgroundColor Black
    } elseif ($UAA_CHECK) {
        Write-Host "✅ You have User Access Administrator" -ForegroundColor Green
    } else {
        Write-Host "❌ No elevated permissions in this subscription" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host ""
}

# Switch back to default
az account set --subscription "d06b42ec-a0ae-4415-b534-e6635ac7e7ed"
Write-Host "Switched back to ennrgyai subscription" -ForegroundColor Cyan

