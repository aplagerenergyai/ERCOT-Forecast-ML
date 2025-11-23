# GitHub Setup Guide

## ✅ Step 1: Create `.gitignore` (COMPLETED)

A `.gitignore` file has been created to protect sensitive data.

## ⚠️ Step 2: BEFORE Your First Commit - Security Check

### Critical: Verify No Sensitive Data Exists

Run these commands to check for sensitive files:

```bash
# Check if .env exists (it should NOT be committed)
ls -la .env

# Check for any credentials
find . -name "*.key" -o -name "*.pem" -o -name "credentials.json"
```

If you find a `.env` file, **DO NOT COMMIT IT**. The `.gitignore` will block it, but double-check.

### What WILL Be Committed ✅

- All Python scripts (`build_features.py`, `train_*.py`, `score.py`, etc.)
- All Azure ML YAML files (`aml_*.yml`)
- Configuration files (`config/settings.json`, `requirements.txt`)
- Docker files (`Dockerfile`, `.dockerignore`)
- Helper scripts (`*.sh`, `*.bat`)
- GitHub Actions workflows (`.github/workflows/*.yml`)
- Documentation (`*.md`)

### What WILL NOT Be Committed 🚫

- `.env` file (database credentials)
- Local model files (`*.pkl`, `*.pt`)
- Local data files (`*.parquet`, `*.csv`)
- Python cache (`__pycache__`)
- Virtual environments (`venv/`)

## 📦 Step 3: Initialize Git and Commit

### If this is a NEW repository:

```bash
# Initialize git
git init

# Add all files (gitignore will protect sensitive ones)
git add .

# Create first commit
git commit -m "Initial commit: ERCOT ML Pipeline - Complete Steps 1-6"

# Create main branch
git branch -M main

# Add your GitHub remote (replace with YOUR repository URL)
git remote add origin https://github.com/YOUR_USERNAME/forecasting-ml.git

# Push to GitHub
git push -u origin main
```

### If this repository ALREADY EXISTS on GitHub:

```bash
# Add all files
git add .

# Commit
git commit -m "Complete ERCOT ML Pipeline - Steps 1-6 with automation"

# Push
git push origin main
```

## 🔧 Step 4: Configure GitHub Secrets for CI/CD

Your GitHub Actions workflow (`.github/workflows/aml_ci_cd.yml`) requires Azure credentials.

### 4.1: Create Azure Service Principal

Run this in Azure CLI:

```bash
az ad sp create-for-rbac --name "github-actions-ercot-ml" \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/<YOUR_RESOURCE_GROUP> \
  --sdk-auth
```

This will output a JSON object like:

```json
{
  "clientId": "...",
  "clientSecret": "...",
  "subscriptionId": "...",
  "tenantId": "...",
  "activeDirectoryEndpointUrl": "...",
  "resourceManagerEndpointUrl": "...",
  "activeDirectoryGraphResourceId": "...",
  "sqlManagementEndpointUrl": "...",
  "galleryEndpointUrl": "...",
  "managementEndpointUrl": "..."
}
```

**Copy this entire JSON output.**

### 4.2: Add GitHub Secret

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `AZURE_CREDENTIALS`
5. Value: Paste the entire JSON from step 4.1
6. Click **Add secret**

### 4.3: Update Workflow File

Before the workflow can run, you MUST update these placeholders in `.github/workflows/aml_ci_cd.yml`:

```yaml
env:
  AZURE_CONTAINER_REGISTRY: <your-acr-name>.azurecr.io      # Replace with YOUR ACR name
  AZURE_ML_WORKSPACE: <your-aml-workspace-name>             # Replace with YOUR workspace name
  AZURE_ML_RESOURCE_GROUP: <your-aml-resource-group>        # Replace with YOUR resource group
  AZURE_ML_SUBSCRIPTION_ID: <your-azure-subscription-id>    # Replace with YOUR subscription ID
```

## 🚀 Step 5: What Happens After First Push

Once you push to `main` branch, GitHub Actions will automatically:

1. ✅ Build Docker image
2. ✅ Log in to Azure Container Registry
3. ✅ Push image to ACR
4. ✅ Tag image as `latest`
5. ✅ Trigger `aml_full_pipeline.yml` in Azure ML

## 🔄 Step 6: Set Up Pull Request Workflow

When you work on a new feature:

```bash
# Create a feature branch
git checkout -b feature/my-new-feature

# Make changes
# ... edit files ...

# Commit
git add .
git commit -m "Add new feature"

# Push to GitHub
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub. The CI/CD workflow will:
- ✅ Run linting
- ✅ Run unit tests
- ✅ Build container (but NOT push to ACR)

## 📋 Recommended Git Workflow

### Daily Work:

```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/update-model-hyperparameters

# Make changes, test locally
# ... work ...

# Commit frequently
git add .
git commit -m "Tune LightGBM hyperparameters"

# Push to GitHub
git push origin feature/update-model-hyperparameters

# Create Pull Request on GitHub
# Review → Merge to main
```

### After Merge:

```bash
# Switch back to main
git checkout main

# Pull merged changes
git pull origin main
```

## 🛡️ Security Best Practices

### ✅ DO:
- Commit all code, YAML, configs
- Use environment variables for secrets
- Store credentials in Azure Key Vault
- Use GitHub Secrets for CI/CD
- Review `.gitignore` regularly

### 🚫 DO NOT:
- Commit `.env` files
- Commit API keys or passwords
- Commit large data files (> 50 MB)
- Commit trained models (store in Azure ML)
- Commit database connection strings

## 📊 Monitoring Your Pipeline

After pushing to GitHub:

1. **Check GitHub Actions**: Go to **Actions** tab in your repository
2. **Check Azure ML**: Go to Azure ML Studio → Jobs
3. **Check ACR**: Verify Docker image was pushed
4. **Check Logs**: Review pipeline logs in Azure ML

## ❓ Troubleshooting

### Issue: "Permission denied" when pushing

**Solution**: Check if you have write access to the repository.

```bash
# Verify remote URL
git remote -v

# If using HTTPS, you may need to authenticate
git push origin main
```

### Issue: GitHub Actions failing

**Solution**: Check:
1. Is `AZURE_CREDENTIALS` secret configured?
2. Are placeholder values in `aml_ci_cd.yml` replaced?
3. Does the Azure service principal have correct permissions?

### Issue: Large files rejected

**Solution**: If you accidentally try to commit large files:

```bash
# Remove from staging
git reset HEAD large_file.parquet

# Ensure it's in .gitignore
echo "large_file.parquet" >> .gitignore
```

## 📝 Summary

**Before you commit:**
1. ✅ Verify `.gitignore` exists
2. ✅ Check for `.env` files (should NOT exist in repo)
3. ✅ Review what will be committed: `git status`

**First commit:**
```bash
git add .
git commit -m "Initial commit: ERCOT ML Pipeline"
git push origin main
```

**Configure Azure:**
1. Create service principal
2. Add `AZURE_CREDENTIALS` to GitHub secrets
3. Update workflow placeholders

**You're done!** 🎉

Your pipeline will now automatically:
- Build and deploy on every push to `main`
- Run tests on every pull request
- Execute scheduled jobs in Azure ML

