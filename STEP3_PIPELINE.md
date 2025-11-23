# Step 3: Azure ML Training Pipeline

## Overview

Step 3 orchestrates the parallel training of all three ML models (LightGBM, XGBoost, Deep Learning) using a single Azure ML pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure ML Pipeline                        │
│              aml_training_pipeline.yml                      │
│                                                             │
│  Input: workspaceblobstore/paths/features/                 │
│         └── hourly_features.parquet                        │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │           Parallel Job Execution                   │   │
│  │                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│   │
│  │  │ train_lgbm   │  │ train_xgb    │  │train_deep││   │
│  │  │ (cpu-cluster)│  │ (cpu-cluster)│  │(gpu-cluster)│  │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬────┘│   │
│  └─────────│──────────────────│─────────────────│─────┘   │
│            │                  │                 │         │
│  Outputs: ↓                  ↓                 ↓         │
│   • models/lgbm/lgbm_model.pkl                           │
│   • models/xgb/xgb_model.pkl                             │
│   • models/deep/deep_model.pt                            │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline Structure

### Input
- **Source**: `azureml://datastores/workspaceblobstore/paths/features/`
- **File**: `hourly_features.parquet`
- **Shared by**: All three training jobs

### Jobs (Parallel Execution)

#### 1. train_lightgbm
- **Script**: `train_lgbm.py`
- **Compute**: `cpu-cluster`
- **Environment**: `azureml:py310@latest`
- **Output**: `lgbm_model_output` → `models/lgbm/lgbm_model.pkl`

#### 2. train_xgboost
- **Script**: `train_xgb.py`
- **Compute**: `cpu-cluster`
- **Environment**: `azureml:py310@latest`
- **Output**: `xgb_model_output` → `models/xgb/xgb_model.pkl`

#### 3. train_deep
- **Script**: `train_deep.py`
- **Compute**: `gpu-cluster`
- **Environment**: `azureml:py310@latest`
- **Output**: `deep_model_output` → `models/deep/deep_model.pt`

### Outputs
All outputs are written to workspaceblobstore:
```
workspaceblobstore/
└── paths/
    └── models/
        ├── lgbm/
        │   └── lgbm_model.pkl
        ├── xgb/
        │   └── xgb_model.pkl
        └── deep/
            └── deep_model.pt
```

## File Structure

### Pipeline Definition
- **`aml_training_pipeline.yml`**: Main pipeline orchestrator

### Individual Job Definitions
- **`aml_train_lgbm.yml`**: LightGBM standalone job
- **`aml_train_xgb.yml`**: XGBoost standalone job
- **`aml_train_deep.yml`**: Deep Learning standalone job

### Training Scripts
- **`train_lgbm.py`**: LightGBM training logic
- **`train_xgb.py`**: XGBoost training logic
- **`train_deep.py`**: Deep Learning (LSTM) training logic

### Supporting Modules
- **`dataloader.py`**: Data loading and preprocessing
- **`metrics.py`**: Evaluation metrics

## How to Run

### Prerequisites

1. **Features must be built first**:
   ```bash
   az ml job create --file aml_build_features.yml --web
   ```
   
   Wait until complete and verify:
   ```bash
   az ml data list --workspace-name <your-workspace>
   ```

2. **Compute clusters must exist**:
   - `cpu-cluster` (for LightGBM and XGBoost)
   - `gpu-cluster` (for Deep Learning)

### Run the Full Pipeline

```bash
az ml job create --file aml_training_pipeline.yml --web
```

This will:
1. Create a pipeline job with 3 parallel steps
2. Each step reads from the same features folder
3. Each step writes to its own model folder
4. All steps run simultaneously (no dependencies between them)

### Monitor Progress

```bash
# View pipeline status
az ml job show --name <job-name>

# Stream logs from a specific job
az ml job stream --name <job-name>
```

Or use Azure ML Studio:
- Navigate to: Portal → Azure ML Workspace → Jobs
- Click on the pipeline run
- View the graph showing all three parallel jobs
- Click each job to see logs and metrics

## Expected Runtime

| Job              | Compute      | Estimated Time |
|------------------|--------------|----------------|
| train_lightgbm   | cpu-cluster  | 10-20 minutes  |
| train_xgboost    | cpu-cluster  | 10-20 minutes  |
| train_deep       | gpu-cluster  | 30-60 minutes  |
| **Total (parallel)** | **-**    | **30-60 minutes** |

Because jobs run in parallel, total pipeline time = max(job times) ≈ 30-60 minutes.

## Environment Variables

Each training script automatically receives:

### Inputs
```python
# Features input path
features_path = os.environ["AZUREML_INPUT_features"]
# → /mnt/batch/tasks/.../workspaceblobstore/features/
```

### Outputs
```python
# Model output path
output_path = os.environ["AZUREML_OUTPUT_model"]
# → /mnt/batch/tasks/.../workspaceblobstore/models/<modelname>/
```

These are automatically set by Azure ML based on the pipeline YAML configuration.

## Pipeline YAML Breakdown

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: ERCOT_DART_Training_Pipeline

# Shared input for all jobs
inputs:
  features_input:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/features/

# Pipeline-level outputs
outputs:
  lgbm_model_output:
    type: uri_folder
  xgb_model_output:
    type: uri_folder
  deep_model_output:
    type: uri_folder

# Three parallel jobs
jobs:
  train_lightgbm:
    type: command
    command: python train_lgbm.py
    compute: cpu-cluster
    inputs:
      features: ${{parent.inputs.features_input}}
    outputs:
      model: ${{parent.outputs.lgbm_model_output}}

  train_xgboost:
    # Similar structure...

  train_deep:
    # Similar structure with gpu-cluster...
```

### Key Features
- **`${{parent.inputs.features_input}}`**: References pipeline-level input
- **`${{parent.outputs.lgbm_model_output}}`**: References pipeline-level output
- **No dependencies between jobs**: All run in parallel
- **Automatic output registration**: Models are automatically tracked

## Running Individual Jobs

You can also run each model training separately:

```bash
# Train only LightGBM
az ml job create --file aml_train_lgbm.yml --web

# Train only XGBoost
az ml job create --file aml_train_xgb.yml --web

# Train only Deep Learning
az ml job create --file aml_train_deep.yml --web
```

This is useful for:
- Testing individual models
- Re-training a specific model
- Debugging issues with one model type

## Output Model Format

Each model is saved as a pickled dictionary containing:

### LightGBM and XGBoost (*.pkl)
```python
{
    'model': <trained_model>,
    'feature_columns': [...],
    'scaler': StandardScaler(...),
    'categorical_encoders': {...},
    'metrics': {
        'train': {'rmse': ..., 'mae': ..., 'mape': ..., 'r2': ...},
        'val': {...},
        'test': {...}
    }
}
```

### Deep Learning (*.pt)
```python
{
    'model_state_dict': <torch_state_dict>,
    'input_dim': 50,
    'feature_columns': [...],
    'scaler': StandardScaler(...),
    'categorical_encoders': {...},
    'metrics': {...}
}
```

## Loading Trained Models

### LightGBM/XGBoost
```python
import pickle

with open('lgbm_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
scaler = model_dict['scaler']
feature_columns = model_dict['feature_columns']

# Make predictions
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

### Deep Learning
```python
import torch

checkpoint = torch.load('deep_model.pt')

# Recreate model architecture
model = LSTMRegressor(
    input_dim=checkpoint['input_dim'],
    hidden_dim=128,
    num_layers=2
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
X_tensor = torch.FloatTensor(X_new_scaled)
predictions = model(X_tensor).detach().numpy()
```

## Troubleshooting

### Issue: Pipeline fails with "Input not found"
**Cause**: Features haven't been built yet
**Solution**: Run Step 1 first:
```bash
az ml job create --file aml_build_features.yml --web
```

### Issue: "gpu-cluster not found"
**Cause**: GPU cluster doesn't exist
**Solution**: Either create GPU cluster or modify `train_deep.yml` to use `cpu-cluster` (slower)

### Issue: One job fails but others succeed
**Cause**: Jobs run independently, so failures don't cascade
**Solution**: 
- Check logs for the failed job
- Fix the issue
- Re-run just that job using the individual YAML

### Issue: "ModuleNotFoundError: No module named 'category_encoders'"
**Cause**: Missing dependency
**Solution**: Update environment.yml or use a custom environment with all dependencies

### Issue: Models training on same data produce different results
**Cause**: This is expected - different algorithms, different random seeds
**Solution**: Compare metrics and select the best performer

## Performance Comparison

After pipeline completes, compare models:

```python
import pickle

# Load all models
with open('models/lgbm/lgbm_model.pkl', 'rb') as f:
    lgbm = pickle.load(f)
with open('models/xgb/xgb_model.pkl', 'rb') as f:
    xgb = pickle.load(f)

# Compare test metrics
print("LightGBM Test RMSE:", lgbm['metrics']['test']['rmse'])
print("XGBoost Test RMSE:", xgb['metrics']['test']['rmse'])
```

## Next Steps After Pipeline Completes

1. **Download Models**:
   ```bash
   az ml job download --name <job-name> --output-name lgbm_model_output
   ```

2. **Register Best Model**:
   ```bash
   az ml model create --name ercot-dart-predictor \
                      --path models/lgbm/lgbm_model.pkl \
                      --type custom_model
   ```

3. **Deploy to Endpoint**:
   ```bash
   az ml online-endpoint create --name ercot-dart-endpoint
   az ml online-deployment create --endpoint ercot-dart-endpoint \
                                  --model ercot-dart-predictor:1
   ```

4. **Set Up Monitoring**:
   - Data drift detection
   - Model performance tracking
   - Automated retraining triggers

## Cost Optimization

To reduce costs:

1. **Use spot instances**: Lower priority, cheaper
2. **Right-size clusters**: Match VM size to workload
3. **Auto-scale**: Scale down when idle
4. **Delete after completion**: Remove compute when not in use

```bash
# Auto-delete compute after job
az ml compute create --name cpu-cluster \
                     --type amlcompute \
                     --min-instances 0 \
                     --max-instances 4 \
                     --idle-time-before-scale-down 300
```

## Summary

✅ **One command runs all three models in parallel**  
✅ **Shared input (features) for consistency**  
✅ **Independent outputs for each model**  
✅ **Full logging and metrics for all models**  
✅ **Automatic model registration in workspace**  
✅ **Ready for comparison and deployment**  

---

**Run the pipeline**:
```bash
az ml job create --file aml_training_pipeline.yml --web
```

