# Pre-Commit Checklist ✅

## Before Your First `git push`, verify:

### 🔒 Security (CRITICAL)

- [ ] `.gitignore` file exists
- [ ] No `.env` file in repository (run: `ls -la .env` - should show "not found")
- [ ] No database passwords in any Python files
- [ ] No API keys hardcoded anywhere
- [ ] `config/settings.json` contains NO sensitive data (only paths/flags)

### 📦 Files to Commit (Should be present)

- [ ] Python scripts: `build_features.py`, `train_*.py`, `score.py`, `publish_predictions.py`
- [ ] Azure ML YAMLs: `aml_*.yml`
- [ ] Config: `config/settings.json`, `requirements.txt`
- [ ] Docker: `Dockerfile`, `.dockerignore`
- [ ] Scripts: `*.sh`, `*.bat`
- [ ] Workflows: `.github/workflows/aml_ci_cd.yml`
- [ ] Docs: `*.md`
- [ ] `.gitignore`

### 🚫 Files to NOT Commit (Should be excluded by .gitignore)

- [ ] `.env` file
- [ ] `__pycache__/` directories
- [ ] `*.pkl`, `*.pt`, `*.pth` (model files)
- [ ] `*.parquet`, `*.csv` (data files)
- [ ] `venv/` or `env/` directories
- [ ] Any files with "secret", "key", "credential" in the name

### 🔧 GitHub Configuration (Do AFTER first push)

- [ ] GitHub repository created
- [ ] Azure service principal created (see `GITHUB_SETUP.md`)
- [ ] `AZURE_CREDENTIALS` secret added to GitHub
- [ ] Placeholders in `.github/workflows/aml_ci_cd.yml` replaced with YOUR values:
  - [ ] `AZURE_CONTAINER_REGISTRY`
  - [ ] `AZURE_ML_WORKSPACE`
  - [ ] `AZURE_ML_RESOURCE_GROUP`
  - [ ] `AZURE_ML_SUBSCRIPTION_ID`

## Quick Commands

```bash
# Check what will be committed
git status

# Check for sensitive files
ls -la .env
find . -name "*.key" -o -name "*.pem"

# Commit safely
git add .
git commit -m "Initial commit: ERCOT ML Pipeline"
git push origin main
```

## ✅ You're Ready!

Once all boxes are checked, you can safely commit to GitHub.

