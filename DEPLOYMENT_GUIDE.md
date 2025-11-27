# ERCOT ML Pipeline - Complete Deployment Guide

## 🎯 Overview

This guide covers the complete deployment of the ERCOT DART prediction ML pipeline from development to production.

---

## 📋 Deployment Stages

### Stage 1: Local Development ✅
```bash
# Test feature engineering
python build_features.py

# Test model training
python train_lgbm.py

# Test inference
python score.py
```

### Stage 2: Containerization ✅
```bash
# Build container
make build

# Run inference server
make run

# Test endpoints
make test
```

### Stage 3: Azure ML Training 🚀
```bash
# Build features in cloud
az ml job create --file aml_build_features.yml

# Train models in parallel
az ml job create --file aml_training_pipeline.yml
```

### Stage 4: Production Inference 🌐
```bash
# Push to registry
make deploy

# Deploy to Azure Container Apps
az containerapp create \
  --name ercot-inference \
  --image myregistry.azurecr.io/ercot-ml-pipeline:latest
```

---

## 🏗️ Complete Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                             │
│                SQL Server (9 ERCOT Tables)                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│              STEP 1: FEATURE ENGINEERING                     │
│            (Azure ML Job + Container)                        │
│                                                              │
│  build_features.py                                          │
│    → Load 9 tables                                          │
│    → Normalize timestamps                                   │
│    → Resample 5-min → hourly                                │
│    → Merge features                                         │
│    → Save parquet                                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│         workspaceblobstore/features/hourly_features.parquet  │
│         (1-5 GB, ~1M rows, 50+ features)                    │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│            STEP 2+3: MODEL TRAINING PIPELINE                 │
│            (Azure ML Pipeline + 3 Parallel Jobs)             │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐         │
│  │ LightGBM   │  │ XGBoost    │  │ Deep Learning│         │
│  │ (cpu)      │  │ (cpu)      │  │ (gpu)        │         │
│  │ 10-20 min  │  │ 10-20 min  │  │ 30-60 min    │         │
│  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘         │
│        │               │                │                  │
│        └───────────────┴────────────────┘                  │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│              workspaceblobstore/models/                      │
│                                                              │
│  • lgbm/lgbm_model.pkl                                      │
│  • xgb/xgb_model.pkl                                        │
│  • deep/deep_model.pt                                       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│         STEP 4+5: CONTAINERIZED INFERENCE                    │
│         (Docker + FastAPI + Uvicorn)                         │
│                                                              │
│  Docker Container:                                          │
│    → score.py (FastAPI app)                                 │
│    → Model loader                                           │
│    → Preprocessing pipeline                                 │
│    → Endpoints: /health, /score, /model/info                │
│                                                              │
│  Deployment Options:                                        │
│    • Azure Container Apps                                   │
│    • Azure Kubernetes Service                               │
│    • Azure Container Instances                              │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                  PRODUCTION ENDPOINT                         │
│          https://ercot-inference.azurecontainerapps.io       │
│                                                              │
│  Client Application → POST /score → DART Predictions        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Deployment Commands

### Complete End-to-End (First Time)

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with SQL credentials

# 2. Build features (Azure ML)
az ml job create --file aml_build_features.yml --web

# 3. Train models (Azure ML)
az ml job create --file aml_training_pipeline.yml --web

# 4. Build and push container
make build
make tag REGISTRY=myregistry.azurecr.io
make acr-login
make push

# 5. Deploy to Azure Container Apps
az containerapp create \
  --name ercot-inference \
  --resource-group myresourcegroup \
  --image myregistry.azurecr.io/ercot-ml-pipeline:latest \
  --target-port 5001 \
  --ingress external \
  --env-vars MODEL_TYPE=lgbm \
  --cpu 2 --memory 4Gi

# 6. Test production endpoint
ENDPOINT=$(az containerapp show --name ercot-inference \
  --resource-group myresourcegroup \
  --query properties.configuration.ingress.fqdn -o tsv)

curl https://$ENDPOINT/health
```

### Retraining & Redeployment

```bash
# 1. Retrain models (monthly/quarterly)
az ml job create --file aml_training_pipeline.yml --web

# 2. Rebuild container with new models
make build-no-cache
make deploy

# 3. Update Azure Container App
az containerapp update \
  --name ercot-inference \
  --resource-group myresourcegroup \
  --image myregistry.azurecr.io/ercot-ml-pipeline:latest
```

---

## 📊 Deployment Checklist

### Pre-Deployment

- [ ] `.env` file created with SQL credentials
- [ ] Azure ML workspace configured
- [ ] Compute clusters created (cpu-cluster, gpu-cluster)
- [ ] Azure Container Registry created
- [ ] Service principal created (for CI/CD)

### Feature Engineering

- [ ] `aml_build_features.yml` job succeeds
- [ ] `hourly_features.parquet` exists in workspaceblobstore
- [ ] Parquet file has expected row count (~1M+)
- [ ] All 50+ features present

### Model Training

- [ ] `aml_training_pipeline.yml` job succeeds
- [ ] All 3 models trained successfully
- [ ] Test RMSE < $10/MWh for all models
- [ ] Test R² > 0.65 for all models
- [ ] Models saved in workspaceblobstore

### Containerization

- [ ] Dockerfile builds successfully
- [ ] Container runs locally without errors
- [ ] Health endpoint returns 200
- [ ] Scoring endpoint returns predictions
- [ ] Test script passes all checks

### Production Deployment

- [ ] Container pushed to ACR
- [ ] Azure Container App created
- [ ] Endpoint accessible publicly (or within VNet)
- [ ] Load balancing configured
- [ ] Autoscaling enabled
- [ ] Monitoring configured
- [ ] Alerts set up

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Automatically triggers on:
- Push to `main` branch
- Pull request to `main`
- Manual trigger (`workflow_dispatch`)

Pipeline steps:
1. Build Docker image
2. Run security scan (Trivy)
3. Push to Azure Container Registry
4. Test inference endpoint
5. Trigger Azure ML pipeline (optional)
6. Deploy to staging (optional)

### Setup

1. **Create GitHub Secrets**:
   - `ACR_USERNAME`
   - `ACR_PASSWORD`
   - `AZURE_CREDENTIALS`
   - `AZURE_RESOURCE_GROUP`
   - `AZURE_ML_WORKSPACE`

2. **Enable Workflow**:
   ```bash
   # Commit workflow file
   git add .github/workflows/build_and_push.yml
   git commit -m "Add CI/CD pipeline"
   git push origin main
   ```

3. **Monitor**:
   - GitHub Actions tab
   - Azure ML Studio (for training jobs)
   - Azure Portal (for container apps)

---

## 📈 Monitoring & Observability

### Application Insights Integration

Add to Dockerfile:
```dockerfile
ENV APPLICATIONINSIGHTS_CONNECTION_STRING=$APPINSIGHTS_CONN_STRING
RUN pip install opencensus-ext-azure
```

Update score.py:
```python
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(
    connection_string=os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING']
))
```

### Key Metrics to Monitor

- **Inference Metrics**:
  - Request rate (requests/second)
  - Latency (p50, p95, p99)
  - Error rate (%)
  - Prediction distribution

- **Model Metrics**:
  - Model version
  - Feature drift
  - Prediction drift
  - Data quality

- **System Metrics**:
  - CPU utilization
  - Memory usage
  - Disk I/O
  - Network traffic

### Alerts

Set up alerts for:
- Error rate > 5%
- Latency p95 > 1000ms
- CPU > 80% for 5 minutes
- Memory > 90%
- Model drift detected

---

## 🔐 Security Considerations

### 1. API Authentication

Add API key authentication:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/score")
async def score(request: PredictionRequest, api_key: str = Security(api_key_header)):
    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... scoring logic
```

### 2. HTTPS Only

Enable HTTPS in Azure Container Apps:
```bash
az containerapp ingress enable \
  --name ercot-inference \
  --type external \
  --allow-insecure false
```

### 3. Network Isolation

Deploy in VNet:
```bash
az containerapp create \
  --name ercot-inference \
  --environment myenvironment \
  --vnet-name myvnet \
  --subnet mysubnet
```

### 4. Managed Identity

Use managed identity for Azure resources:
```bash
az containerapp identity assign \
  --name ercot-inference \
  --resource-group myresourcegroup \
  --system-assigned
```

---

## 💰 Cost Optimization

### Compute Optimization

| Resource | Standard | Optimized | Savings |
|----------|----------|-----------|---------|
| Training Compute | 4x D4s_v3 (always on) | Spot instances, auto-scale | 70% |
| Inference Compute | 3 replicas (always on) | Autoscale 1-5 replicas | 40% |
| GPU Training | 1x NC6 (always on) | Use on-demand | 60% |

### Storage Optimization

- Use lifecycle policies to archive old parquet files
- Compress models before storage
- Use cool storage for historical data

### Total Estimated Monthly Cost

| Component | Monthly Cost |
|-----------|--------------|
| Azure ML Compute (training) | $100-200 |
| Container Apps (inference) | $50-150 |
| Storage (blobs) | $5-20 |
| Container Registry | $5 |
| Networking | $10-30 |
| **Total** | **$170-405/month** |

*(Costs vary based on usage, region, and spot instance availability)*

---

## 🐛 Common Issues & Solutions

### Issue: Inference is slow

**Symptoms**: Latency > 1000ms

**Solutions**:
- Use LightGBM instead of Deep Learning (5ms vs 15ms)
- Enable model caching
- Add more replicas
- Use faster CPU/GPU

### Issue: Models become stale

**Symptoms**: Prediction accuracy drops over time

**Solutions**:
- Schedule monthly retraining
- Monitor prediction drift
- Set up automated retraining pipeline
- Use online learning (if applicable)

### Issue: High costs

**Symptoms**: Monthly Azure bill > $500

**Solutions**:
- Use spot instances for training
- Enable autoscaling (scale to zero when idle)
- Archive old data to cool storage
- Optimize container image size

### Issue: Container crashes

**Symptoms**: Container repeatedly restarts

**Solutions**:
- Check logs: `docker logs ercot-ml-pipeline`
- Increase memory limit
- Verify model files are accessible
- Check environment variables

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| **QUICK_START.md** | Fast execution guide |
| **PIPELINE_GUIDE.md** | Complete pipeline documentation |
| **PROJECT_STRUCTURE.md** | Architecture overview |
| **STEP2_SUMMARY.md** | Training implementation |
| **STEP3_PIPELINE.md** | Pipeline orchestration |
| **STEP5_CONTAINERIZATION.md** | Container deployment |
| **DEPLOYMENT_GUIDE.md** | This document |
| **COMPLETE_PIPELINE_SUMMARY.md** | Master summary |

---

## ✅ Production Readiness Checklist

### Code Quality
- [ ] All linter errors resolved
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] Code reviewed

### Performance
- [ ] Inference latency < 100ms
- [ ] Throughput > 100 req/s
- [ ] Model accuracy meets requirements
- [ ] Load testing completed

### Reliability
- [ ] Health checks implemented
- [ ] Error handling complete
- [ ] Logging configured
- [ ] Monitoring enabled
- [ ] Alerts configured

### Security
- [ ] Secrets in environment variables
- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Vulnerability scan clean
- [ ] RBAC configured

### Operations
- [ ] CI/CD pipeline working
- [ ] Backup strategy defined
- [ ] Disaster recovery plan
- [ ] Runbook documented
- [ ] On-call rotation setup

---

## 🎓 Training & Knowledge Transfer

### For Data Scientists

1. **Model Development**: Use Jupyter notebooks for experimentation
2. **Feature Engineering**: Update `build_features.py` for new features
3. **Model Training**: Modify hyperparameters in training scripts
4. **Evaluation**: Use metrics.py for consistent evaluation

### For ML Engineers

1. **Pipeline**: Understand `aml_training_pipeline.yml` structure
2. **Containerization**: Know Docker and Dockerfile
3. **Deployment**: Understand Azure Container Apps
4. **Monitoring**: Set up Application Insights

### For DevOps

1. **CI/CD**: GitHub Actions workflow
2. **Infrastructure**: Azure resources (AML, ACR, ACA)
3. **Monitoring**: Application Insights, Azure Monitor
4. **Troubleshooting**: Logs, metrics, alerts

---

## 📞 Support & Escalation

### Tier 1: Self-Service
- Check documentation
- Review logs
- Test health endpoint
- Run test script

### Tier 2: Team Support
- Slack channel: #ercot-ml-support
- Email: ml-team@company.com
- Office hours: Mon-Fri 9am-5pm

### Tier 3: On-Call
- PagerDuty: ERCOT ML Oncall
- Phone: +1-xxx-xxx-xxxx
- Escalation: CTO

---

## 🎯 Roadmap

### Q1 2025
- [ ] Deploy to production
- [ ] Set up monitoring
- [ ] Establish retraining schedule
- [ ] Document operational procedures

### Q2 2025
- [ ] Add feature drift detection
- [ ] Implement A/B testing
- [ ] Optimize inference performance
- [ ] Add more settlement points

### Q3 2025
- [ ] Multi-model ensemble
- [ ] Real-time feature updates
- [ ] Advanced monitoring dashboard
- [ ] Automated model selection

### Q4 2025
- [ ] Geographic expansion
- [ ] New forecast horizons
- [ ] Integration with trading systems
- [ ] Advanced anomaly detection

---

**🎊 Deployment guide complete! You're ready for production!**

Start with: `make build && make test && make deploy`

