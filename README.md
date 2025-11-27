# ERCOT Price Forecasting - Azure ML Pipeline

This repository contains production-ready machine learning pipelines for forecasting electricity prices in the ERCOT (Electric Reliability Council of Texas) market using Azure Machine Learning.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Pipeline Steps](#pipeline-steps)
- [Submitting Jobs](#submitting-jobs)
- [Compute Resources](#compute-resources)
- [Output Storage](#output-storage)
- [Extending to Other ISOs](#extending-to-other-isos)

---

## 🏗️ Architecture Overview

The pipeline consists of four main components:

1. **Feature Engineering** (`build_features.py`) - Loads data from SQL Server, resamples to hourly, and creates unified features
2. **LightGBM Training** (`train_lightgbm.py`) - Trains gradient boosting model
3. **XGBoost Training** (`train_xgboost.py`) - Trains XGBoost model with tuned hyperparameters
4. **Deep Learning** (`train_deep.py`) - Trains LSTM neural network for sequence forecasting

Each component can be run independently or as part of an automated Azure ML pipeline.

---

## 📦 Prerequisites

- **Python 3.8+**
- **Azure subscription** with Azure Machine Learning workspace
- **SQL Server** with ERCOT historical data tables
- **ODBC Driver 18 for SQL Server**

---

## 🚀 Installation

### 1. Install Azure ML CLI v2

The Azure ML CLI v2 is the recommended way to interact with Azure Machine Learning.

**Install via Azure CLI extension:**

```bash
# Install Azure CLI (if not already installed)
# Windows: Download from https://aka.ms/installazurecliwindows
# Linux/Mac: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install the Azure ML extension
az extension add --name ml
```

**Verify installation:**

```bash
az ml --version
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas`
- `numpy`
- `sqlalchemy`
- `pyodbc`
- `python-dotenv`
- `lightgbm`
- `xgboost`
- `torch`
- `pytorch-lightning`
- `scikit-learn`
- `pyarrow`
- `joblib`

---

## 🔐 Configuration

### 1. Azure Authentication

Authenticate with your Azure account:

```bash
az login
```

This will open a browser window for authentication. After successful login, set your default subscription:

```bash
az account set --subscription <subscription-id>
```

Set your Azure ML workspace as the default:

```bash
az configure --defaults workspace=<workspace-name> group=<resource-group-name>
```

### 2. Environment Variables

Create a `.env` file in the project root with your SQL Server connection details:

```env
# SQL Server Configuration
SQL_SERVER=your-server.database.windows.net
SQL_DATABASE=your-database-name
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
```

**Required Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `SQL_SERVER` | SQL Server hostname | `myserver.database.windows.net` |
| `SQL_DATABASE` | Database name | `ERCOTData` |
| `SQL_USERNAME` | Database username | `admin` |
| `SQL_PASSWORD` | Database password | `SecureP@ssw0rd!` |

**⚠️ Security Note:** Never commit the `.env` file to version control. Add it to `.gitignore`.

---

## 📊 Pipeline Steps

### Step 1: Build Features

Loads ERCOT data from SQL Server tables and creates hourly feature dataset.

**Tables loaded:**
- `[ERCOT].[hist_LMPbyResourceNodesLoadZonesandTradingHubs]`
- `[ERCOT].[hist_SCEDShadowPricesandBindingTransmissionConstraints]`
- `[ERCOT].[hist_SolarPowerProductionActual5MinuteAveragedValues]`
- `[ERCOT].[hist_SolarPowerProductionHourlyAveragedActualandForecastedValues]`
- `[ERCOT].[hist_WindPowerProductionHourlyAveragedActualandForecastedValues]`

**Output:** `features.parquet` (unified hourly dataset)

### Step 2: Train LightGBM

Trains a LightGBM gradient boosting model with default parameters.

**Output:** `model_output/lightgbm_model.pkl`

### Step 3: Train XGBoost

Trains an XGBoost model with tuned hyperparameters:
- `n_estimators = 800`
- `max_depth = 8`
- `learning_rate = 0.05`

**Output:** `model_output/xgboost_model.pkl`

### Step 4: Train Deep Learning Model

Trains a 2-layer LSTM neural network for time-series forecasting:
- Input sequence: 168 hours (7 days)
- Forecast horizon: 24 hours ahead
- GPU-accelerated training

**Outputs:** 
- `model_output/deep_model.pt`
- `model_output/feature_scaler.pkl`
- `model_output/target_scaler.pkl`

---

## 🎯 Submitting Jobs

### Submit Individual Jobs

Submit each job to Azure ML using the CLI:

```bash
# 1. Build features (runs on CPU cluster)
az ml job create --file aml_build_features.yml

# 2. Train LightGBM (runs on CPU cluster)
az ml job create --file aml_lightgbm.yml

# 3. Train XGBoost (runs on CPU cluster)
az ml job create --file aml_xgboost.yml

# 4. Train Deep Learning Model (runs on GPU cluster)
az ml job create --file aml_deep.yml
```

### Monitor Job Progress

```bash
# List recent jobs
az ml job list --max-results 10

# Get job details
az ml job show --name <job-name>

# Stream job logs
az ml job stream --name <job-name>
```

### Download Job Outputs

```bash
# Download all outputs
az ml job download --name <job-name> --output-name outputs --download-path ./local_outputs

# Download model outputs
az ml job download --name <job-name> --output-name model_output --download-path ./models
```

---

## 💻 Compute Resources

### CPU Cluster

**Name:** `cpu-cluster`

**Used for:**
- Feature engineering (`build_features.py`)
- LightGBM training (`train_lightgbm.py`)
- XGBoost training (`train_xgboost.py`)

**Recommended configuration:**
- VM Size: `Standard_D4s_v3` or higher
- Min nodes: 0 (auto-scale down when idle)
- Max nodes: 4
- Idle time before scale down: 120 seconds

### GPU Cluster

**Name:** `gpu-cluster`

**Used for:**
- Deep learning training (`train_deep.py`)

**Recommended configuration:**
- VM Size: `Standard_NC6` or `Standard_NC6s_v3` (with NVIDIA Tesla V100)
- Min nodes: 0 (auto-scale down when idle)
- Max nodes: 2
- Idle time before scale down: 300 seconds

### Creating Compute Clusters

If clusters don't exist, create them:

```bash
# Create CPU cluster
az ml compute create --name cpu-cluster \
  --type amlcompute \
  --size Standard_D4s_v3 \
  --min-instances 0 \
  --max-instances 4

# Create GPU cluster
az ml compute create --name gpu-cluster \
  --type amlcompute \
  --size Standard_NC6 \
  --min-instances 0 \
  --max-instances 2
```

---

## 📁 Output Storage

### Azure ML Storage Structure

All outputs are stored in your Azure ML workspace's default blob storage:

```
azureml://datastores/workspaceblobstore/paths/
├── features/
│   └── features.parquet          # Feature engineering output
├── models/
│   ├── lightgbm_model.pkl        # LightGBM model
│   ├── xgboost_model.pkl         # XGBoost model
│   ├── deep_model.pt             # LSTM model weights
│   ├── feature_scaler.pkl        # Feature normalization scaler
│   └── target_scaler.pkl         # Target normalization scaler
└── logs/                         # Job execution logs
```

### Accessing Outputs

**Via Azure ML Studio:**
1. Navigate to your workspace in the Azure Portal
2. Go to "Jobs" → Select your job
3. Click "Outputs + logs" tab
4. Browse or download files

**Via CLI:**
```bash
# List outputs
az ml job show --name <job-name> --query outputs

# Download specific output
az ml job download --name <job-name> --output-name model_output
```

**Via Python SDK:**
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential())
job = ml_client.jobs.get("<job-name>")
ml_client.jobs.download(job.name, download_path="./outputs")
```

---

## 🌍 Extending to Other ISOs

This pipeline architecture is designed to be ISO-agnostic. To extend it to other electricity markets (e.g., PJM, CAISO, MISO, SPP):

### 1. Update Data Source

Modify `build_features.py` to load data from the new ISO's tables:

```python
def process_iso_lmp_data(engine, iso_name: str) -> pd.DataFrame:
    """
    Generalized function to load LMP data for any ISO.
    
    Args:
        engine: SQLAlchemy engine
        iso_name: ISO identifier ('ERCOT', 'PJM', 'CAISO', etc.)
    """
    table_name = f"[{iso_name}].[hist_LMP]"
    df = load_table(engine, table_name)
    # ... processing logic
    return df_hourly
```

### 2. Parameterize Configuration

Create ISO-specific configuration files:

**ercot_config.yml:**
```yaml
iso: ERCOT
tables:
  lmp: "[ERCOT].[hist_LMPbyResourceNodesLoadZonesandTradingHubs]"
  wind: "[ERCOT].[hist_WindPowerProductionHourlyAveragedActualandForecastedValues]"
  solar: "[ERCOT].[hist_SolarPowerProductionHourlyAveragedActualandForecastedValues]"
```

**pjm_config.yml:**
```yaml
iso: PJM
tables:
  lmp: "[PJM].[hist_LMP]"
  generation: "[PJM].[hist_Generation]"
  load: "[PJM].[hist_Load]"
```

### 3. Update Pipeline Scripts

Add `--iso` parameter to all scripts:

```bash
python build_features.py --iso ERCOT --output_path outputs/ercot_features.parquet
python build_features.py --iso PJM --output_path outputs/pjm_features.parquet
```

### 4. Create ISO-Specific Azure ML Jobs

**aml_pjm_lightgbm.yml:**
```yaml
display_name: train_lightgbm_pjm
command: >
  python train_lightgbm.py --data_path ./inputs/pjm_features/features.parquet
```

### 5. Key Extensibility Points

The following components are designed to be reusable across ISOs:

✅ **Feature Engineering Logic** - Timestamp detection, resampling, merging  
✅ **Model Training Scripts** - ISO-agnostic, work with any tabular data  
✅ **Azure ML Infrastructure** - Same compute clusters, environments  
✅ **Evaluation Metrics** - MAE, RMSE, MAPE work for all markets  

**What needs to change:**
- SQL table names and schemas
- Feature column names (can be mapped)
- Market-specific features (e.g., congestion zones differ by ISO)
- Data frequency (some ISOs report 5-min, others 15-min)

### 6. Multi-ISO Pipeline Example

Run parallel pipelines for multiple ISOs:

```bash
# ERCOT Pipeline
az ml job create --file aml_ercot_build_features.yml
az ml job create --file aml_ercot_lightgbm.yml

# PJM Pipeline
az ml job create --file aml_pjm_build_features.yml
az ml job create --file aml_pjm_lightgbm.yml

# CAISO Pipeline
az ml job create --file aml_caiso_build_features.yml
az ml job create --file aml_caiso_lightgbm.yml
```

---

## 📈 Model Performance

Track model performance across training runs:

```bash
# Compare runs in Azure ML Studio
az ml job list --output table

# Get metrics for specific job
az ml job show --name <job-name> --query metrics
```

**Metrics tracked:**
- **MAE** (Mean Absolute Error) - Average prediction error in $/MWh
- **RMSE** (Root Mean Squared Error) - Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error) - Percentage error metric

---

## 🔧 Troubleshooting

### Common Issues

**1. SQL Connection Errors**
```
Error: Cannot connect to SQL Server
```
- Verify `.env` file exists and contains correct credentials
- Check firewall rules allow connection from Azure ML
- Confirm ODBC Driver 18 is installed

**2. Compute Cluster Not Found**
```
Error: Compute target 'cpu-cluster' not found
```
- Create the compute cluster (see [Compute Resources](#compute-resources))
- Or update YAML files to use an existing cluster name

**3. Environment Not Found**
```
Error: Environment 'forecast-env' not found
```
- Create the environment in Azure ML Studio
- Or reference a different environment in the YAML files

**4. GPU Out of Memory**
```
RuntimeError: CUDA out of memory
```
- Reduce `batch_size` in `train_deep.py`
- Use a larger GPU VM size
- Enable gradient checkpointing (advanced)

---

## 📚 Additional Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure ML CLI v2 Reference](https://docs.microsoft.com/cli/azure/ml)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📝 License

Copyright © 2025 - ERCOT Price Forecasting Project

---

## 👥 Support

For questions or issues:
- Review the troubleshooting section above
- Check Azure ML job logs: `az ml job stream --name <job-name>`
- Consult Azure ML documentation

---

**Last Updated:** November 2025

