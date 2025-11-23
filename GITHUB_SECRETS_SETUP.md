# GitHub Secrets Setup Guide 🔐

## ✅ What You've Already Done:
1. ✅ Created ACR: `ercotforecasting.azurecr.io`
2. ✅ Granted Azure ML workspace access to ACR
3. ✅ Updated `.github/workflows/aml_ci_cd.yml` with ACR name

---

## 🎯 Next Steps: Add GitHub Secrets

You need to add **5 secrets** to your GitHub repository for the CI/CD workflow to work.

### Step 1: Get Your Azure Information

Run these commands in Azure CLI to get the values you need:

```bash
# 1. Get your ACR credentials
az acr credential show --name ercotforecasting

# Output will show:
# "username": "ercotforecasting"
# "passwords": [{ "value": "SOME_LONG_STRING" }]

# 2. Get your ML workspace details
az ml workspace show --name <your-ml-workspace-name> --resource-group <your-resource-group>

# Note down:
# - name (workspace name)
# - resourceGroup
# - location

# 3. Get your subscription ID
az account show --query id -o tsv
```

### Step 2: Create Azure Service Principal

This gives GitHub Actions permission to trigger Azure ML pipelines:

```bash
# Replace with YOUR values
SUBSCRIPTION_ID="<your-subscription-id>"
RESOURCE_GROUP="<your-resource-group>"

# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-ercot-ml" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth
```

**IMPORTANT:** This will output a JSON object. **COPY THE ENTIRE JSON OUTPUT.** It looks like this:

```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

---

## 🔑 Step 3: Add Secrets to GitHub

### Go to your GitHub repository:
1. Click **Settings** (at the top)
2. Click **Secrets and variables** → **Actions** (in left sidebar)
3. Click **New repository secret**

### Add these 5 secrets:

#### Secret #1: `ACR_USERNAME`
- **Value:** `ercotforecasting` (from step 1 above)

#### Secret #2: `ACR_PASSWORD`
- **Value:** The password from `az acr credential show` command (from step 1)
- It's the long string under `"passwords": [{ "value": "..." }]`

#### Secret #3: `AZURE_CREDENTIALS`
- **Value:** The **ENTIRE JSON** from the service principal creation (step 2)
- Paste the whole JSON object, including the curly braces `{}`

#### Secret #4: `AZURE_ML_WORKSPACE`
- **Value:** Your ML workspace name
- Example: `ercot-ml-workspace`

#### Secret #5: `AZURE_RESOURCE_GROUP`
- **Value:** Your resource group name
- Example: `rg-ercot-forecasting`

#### Secret #6: `AZURE_SUBSCRIPTION_ID`
- **Value:** Your Azure subscription ID (GUID)
- Example: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

---

## 📋 Verification Checklist

After adding all secrets, verify in GitHub:

- [ ] Go to **Settings** → **Secrets and variables** → **Actions**
- [ ] You should see 6 secrets listed:
  - `ACR_USERNAME`
  - `ACR_PASSWORD`
  - `AZURE_CREDENTIALS`
  - `AZURE_ML_WORKSPACE`
  - `AZURE_RESOURCE_GROUP`
  - `AZURE_SUBSCRIPTION_ID`

---

## 🚀 Step 4: Test the Workflow

### Option A: Push to Main (Full Workflow)

```bash
# Make sure all files are committed
git add .
git commit -m "Configure CI/CD pipeline"
git push origin main
```

**What will happen:**
1. ✅ Lint and test code
2. ✅ Build Docker image
3. ✅ Push image to ACR (`ercotforecasting.azurecr.io`)
4. ✅ Trigger Azure ML pipeline (prediction only, no retraining)

### Option B: Trigger Retraining

Add `[retrain]` to your commit message:

```bash
git commit -m "Update model parameters [retrain]"
git push origin main
```

**What will happen:**
- Same as above, but triggers **full retraining** in Azure ML

### Option C: Manual Trigger

1. Go to GitHub → **Actions** tab
2. Click **Azure ML CI/CD Pipeline** workflow
3. Click **Run workflow** dropdown
4. Check/uncheck "Trigger full training pipeline"
5. Click **Run workflow**

---

## 🔍 Monitoring the Workflow

### In GitHub:
1. Go to **Actions** tab
2. Click on the running workflow
3. Click on each job to see logs

### In Azure:
1. Go to **Azure ML Studio**
2. Click **Jobs** in left sidebar
3. You'll see the pipeline running

### Check ACR:
```bash
# List images in your ACR
az acr repository list --name ercotforecasting --output table

# Show tags for your image
az acr repository show-tags --name ercotforecasting --repository ercot-ml-pipeline --output table
```

---

## ⚠️ Troubleshooting

### Error: "No such host ercotforecasting.azurecr.io"
**Solution:** Check that ACR name is correct in workflow file (line 21)

### Error: "Failed to authenticate with Azure"
**Solution:** 
1. Verify `AZURE_CREDENTIALS` secret is correct JSON
2. Verify service principal has `contributor` role
3. Run: `az role assignment list --assignee <clientId>`

### Error: "ACR login failed"
**Solution:**
1. Verify `ACR_USERNAME` = `ercotforecasting`
2. Get fresh password: `az acr credential show --name ercotforecasting`
3. Update `ACR_PASSWORD` secret

### Error: "Workspace not found"
**Solution:** Verify `AZURE_ML_WORKSPACE` and `AZURE_RESOURCE_GROUP` secrets match your actual Azure resources

---

## 📊 What Happens After Successful Push?

1. **GitHub Actions builds Docker image** → `ercotforecasting.azurecr.io/ercot-ml-pipeline:main-abc1234`
2. **Image is pushed to ACR**
3. **Azure ML pipeline is triggered**
4. **Pipeline runs:**
   - Build features
   - Train models (if `[retrain]` in commit message)
   - Generate predictions
   - Publish predictions to blob storage

---

## 🎉 Success Indicators

✅ **In GitHub Actions:** All jobs show green checkmarks  
✅ **In ACR:** Image exists with correct tag  
✅ **In Azure ML Studio:** Pipeline job is running/completed  
✅ **In Blob Storage:** Predictions folder has new parquet files  

---

## 📝 Summary - You're at This Step:

```
✅ 1. Created ACR
✅ 2. Granted ML workspace access to ACR
✅ 3. Updated workflow file
🔄 4. ADD GITHUB SECRETS (YOU ARE HERE)
⏳ 5. Commit and push to GitHub
⏳ 6. Monitor workflow execution
```

### Quick Commands to Copy:

```bash
# Get ACR credentials
az acr credential show --name ercotforecasting

# Create service principal
az ad sp create-for-rbac --name "github-actions-ercot-ml" \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUB_ID>/resourceGroups/<YOUR_RG> \
  --sdk-auth

# Test ACR login (optional)
az acr login --name ercotforecasting

# Commit and push
git add .
git commit -m "Setup CI/CD with ACR integration"
git push origin main
```

Go to GitHub → Settings → Secrets → Actions and add the 6 secrets!

