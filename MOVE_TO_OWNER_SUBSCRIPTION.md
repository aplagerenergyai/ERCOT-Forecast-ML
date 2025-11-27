# Move ACR to Subscription Where You Have Owner Rights

## 🎯 Step 1: Find Which Subscription Has Owner Rights

Run this:

### PowerShell:
```powershell
.\check_all_subscriptions.ps1
```

### Bash:
```bash
bash check_all_subscriptions.sh
```

**Look for the subscription that says: "✅✅✅ YOU HAVE OWNER ROLE"**

---

## 🔀 Two Options:

### **Option A: Create NEW ACR in Owner Subscription (EASIEST - 5 minutes)**

If you have Owner rights in `Azure subscription 1` (b7788659-1f79-4e40-b98a-eea87041561f), just create a NEW ACR there:

```bash
# Switch to the subscription where you have Owner
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f

# Create a resource group (or use existing)
OWNER_RG="rg-ercot-ml"
az group create --name $OWNER_RG --location southcentralus

# Create NEW ACR in this subscription
ACR_NAME="ercotforecastingv2"  # Must be globally unique
az acr create \
  --resource-group $OWNER_RG \
  --name $ACR_NAME \
  --sku Basic \
  --location southcentralus \
  --admin-enabled true

# Get the login server
az acr show --name $ACR_NAME --query loginServer -o tsv

# Get credentials
az acr credential show --name $ACR_NAME
```

**✅ DONE!** Now use this ACR in your GitHub workflow.

---

### **Option B: Move Existing ACR Between Subscriptions (COMPLEX - 20+ minutes)**

Moving resources between subscriptions can be tricky. Here's how:

#### Step 1: Check if ACR can be moved
```bash
# Switch to current subscription
az account set --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed

# Get ACR resource ID
ACR_ID=$(az acr show --name ercotforecasting --resource-group ennrgyai-rg --query id -o tsv)

echo "ACR ID: $ACR_ID"
```

#### Step 2: Validate the move
```bash
# Target subscription and resource group
TARGET_SUBSCRIPTION="b7788659-1f79-4e40-b98a-eea87041561f"
TARGET_RG="rg-ercot-ml"  # Create this first in target subscription

# Validate move (dry run)
az resource move \
  --destination-group $TARGET_RG \
  --destination-subscription-id $TARGET_SUBSCRIPTION \
  --ids $ACR_ID \
  --validate-only
```

#### Step 3: Actually move it
```bash
# Move the ACR
az resource move \
  --destination-group $TARGET_RG \
  --destination-subscription-id $TARGET_SUBSCRIPTION \
  --ids $ACR_ID
```

**⚠️ WARNING:** Moving ACR will:
- Break any existing references to it
- May take 30+ minutes
- Requires Owner in BOTH subscriptions

---

## 🎯 My Recommendation: Option A (Create New ACR)

**Why:**
1. ✅ Faster (5 minutes vs 30+ minutes)
2. ✅ Less risk (no downtime)
3. ✅ Cleaner (fresh start)
4. ✅ Old ACR stays as backup

**Downside:**
- You'll have an old, unused ACR in `ennrgyai` subscription (just delete it later)

---

## 📋 After Creating/Moving ACR:

### Update Your Workflow File

Once you have the new ACR, update `.github/workflows/aml_ci_cd.yml`:

```yaml
env:
  PYTHON_VERSION: '3.10'
  REGISTRY: <new-acr-name>.azurecr.io  # Update this
  IMAGE_NAME: ercot-ml-pipeline
  AZURE_ML_WORKSPACE: ${{ secrets.AZURE_ML_WORKSPACE }}
  AZURE_ML_RESOURCE_GROUP: ${{ secrets.AZURE_RESOURCE_GROUP }}
  AZURE_ML_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

### Grant ML Workspace Access (YOU CAN DO THIS NOW!)

If your ML workspace is in `ennrgyai` subscription and ACR is in `Azure subscription 1`:

```bash
# Switch to the ACR subscription (where you have Owner)
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f

# Get ML workspace identity from OTHER subscription
WORKSPACE_IDENTITY=$(az ml workspace show \
  --name energyaiml \
  --resource-group ennrgyai-rg \
  --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed \
  --query identity.principal_id -o tsv)

# Grant AcrPull (NOW YOU HAVE OWNER RIGHTS!)
az role assignment create \
  --assignee $WORKSPACE_IDENTITY \
  --role AcrPull \
  --scope $(az acr show --name <new-acr-name> --query id -o tsv)
```

**✅ THIS WILL WORK because you have Owner in the ACR subscription!**

---

## 🚀 Complete Setup Commands (Option A - New ACR)

```bash
# 1. Switch to Owner subscription
az account set --subscription b7788659-1f79-4e40-b98a-eea87041561f

# 2. Create resource group (if needed)
az group create --name rg-ercot-ml --location southcentralus

# 3. Create ACR
az acr create \
  --resource-group rg-ercot-ml \
  --name ercotforecastingv2 \
  --sku Basic \
  --location southcentralus \
  --admin-enabled true

# 4. Get workspace identity from ML subscription
WORKSPACE_IDENTITY=$(az ml workspace show \
  --name energyaiml \
  --resource-group ennrgyai-rg \
  --subscription d06b42ec-a0ae-4415-b534-e6635ac7e7ed \
  --query identity.principal_id -o tsv)

# 5. Grant AcrPull permission
az role assignment create \
  --assignee $WORKSPACE_IDENTITY \
  --role AcrPull \
  --scope $(az acr show --name ercotforecastingv2 --query id -o tsv)

# 6. Get ACR credentials for GitHub
az acr credential show --name ercotforecastingv2

# 7. Create service principal in ML subscription (for GitHub Actions)
az ad sp create-for-rbac \
  --name "github-actions-ercot-ml" \
  --role contributor \
  --scopes /subscriptions/d06b42ec-a0ae-4415-b534-e6635ac7e7ed/resourceGroups/ennrgyai-rg \
  --sdk-auth
```

---

## 📊 Updated GitHub Secrets

| Secret Name | New Value |
|------------|-----------|
| `ACR_USERNAME` | `ercotforecastingv2` (or your new ACR name) |
| `ACR_PASSWORD` | From `az acr credential show` |
| `AZURE_ML_WORKSPACE` | `energyaiml` (unchanged) |
| `AZURE_RESOURCE_GROUP` | `ennrgyai-rg` (unchanged) |
| `AZURE_SUBSCRIPTION_ID` | `d06b42ec-a0ae-4415-b534-e6635ac7e7ed` (unchanged) |
| `AZURE_CREDENTIALS` | From service principal creation |

---

## ✅ Success Indicators

After creating new ACR:

```bash
# Test ACR login
az acr login --name ercotforecastingv2

# Verify ML workspace has access
az role assignment list \
  --scope $(az acr show --name ercotforecastingv2 --query id -o tsv) \
  --query "[?roleDefinitionName=='AcrPull']" -o table
```

You should see the ML workspace identity with AcrPull role!

---

## 🎉 Next Steps

1. ✅ Run `check_all_subscriptions.ps1` to find Owner subscription
2. ✅ Create new ACR in that subscription
3. ✅ Grant ML workspace AcrPull permission
4. ✅ Update GitHub workflow with new ACR name
5. ✅ Add GitHub secrets
6. ✅ Push to GitHub and watch it work!

