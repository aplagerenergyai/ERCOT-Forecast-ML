# Windows Quick Start Guide

## 🪟 Running on Windows (No `make` Required!)

Since `make` is not available by default on Windows, I've created **Windows batch files** (`.bat`) and **shell scripts** (`.sh`) that work identically to the Makefile commands.

## ⚡ Quick Command Reference

| What You're Using | Build Command | Run Command | Test Command |
|-------------------|---------------|-------------|--------------|
| **Command Prompt** | `build.bat` | `run.bat` | `test.bat` |
| **PowerShell** | `.\build.bat` | `.\run.bat` | `.\test.bat` |
| **Git Bash** | `bash build.sh` | `bash run.sh` | `bash test.sh` |

**👉 Use the commands for YOUR shell type!**

---

## ✅ Prerequisites

1. **Docker Desktop for Windows** installed and running
2. **Git Bash** or **PowerShell** (both work)
3. **curl** (comes with Windows 10/11)

---

## 🚀 Quick Start Commands

### Option 1: Using Batch Files (Command Prompt)

Open **Command Prompt** (not PowerShell):

```cmd
# Build the Docker image
build.bat

# Run the inference server
run.bat

# Test all endpoints
test.bat

# View logs
logs.bat

# Stop container
stop.bat
```

### Option 1b: Using Batch Files (PowerShell)

In **PowerShell**, you must prefix with `.\`:

```powershell
# Build the Docker image
.\build.bat

# Run the inference server
.\run.bat

# Test all endpoints
.\test.bat

# View logs
.\logs.bat

# Stop container
.\stop.bat
```

### Option 1c: Using Shell Scripts (Git Bash)

In **Git Bash**, use the `.sh` files:

```bash
# Build the Docker image
bash build.sh

# Run the inference server
bash run.sh

# Test all endpoints
bash test.sh

# View logs
bash logs.sh

# Stop container
bash stop.sh
```

### Option 2: Direct Docker Commands

If you prefer to use Docker directly:

```cmd
# Build
docker build -t ercot-ml-pipeline:latest .

# Run
docker run -d --name ercot-ml-pipeline -p 5001:5001 -e MODEL_TYPE=lgbm -v "%CD%\models:/app/models:ro" ercot-ml-pipeline:latest

# Test
curl http://localhost:5001/health

# Logs
docker logs ercot-ml-pipeline

# Stop
docker stop ercot-ml-pipeline
docker rm ercot-ml-pipeline
```

---

## 📝 Step-by-Step Instructions

### 1. Build the Container

Navigate to the project folder, then run the build command based on your shell:

**In Command Prompt:**
```cmd
cd "C:\Users\aepla\Dropbox\QKSS\Active Clients\Softsmiths\ERCOTMLModel\forecasting-ml"
build.bat
```

**In PowerShell:**
```powershell
cd "C:\Users\aepla\Dropbox\QKSS\Active Clients\Softsmiths\ERCOTMLModel\forecasting-ml"
.\build.bat
```

**In Git Bash:**
```bash
cd "/c/Users/aepla/Dropbox/QKSS/Active Clients/Softsmiths/ERCOTMLModel/forecasting-ml"
bash build.sh
```

**Expected output:**
```
========================================
Building ERCOT ML Pipeline Docker Image
========================================

[+] Building 45.2s (12/12) FINISHED
...
========================================
Build Complete!
========================================

Image: ercot-ml-pipeline:latest
Next: run.bat to start the container
```

**Build time**: ~5-10 minutes first time, ~1-2 minutes with cache

---

### 2. Start the Inference Server

**In Command Prompt:**
```cmd
run.bat
```

**In PowerShell:**
```powershell
.\run.bat
```

**In Git Bash:**
```bash
bash run.sh
```

**Expected output:**
```
========================================
Starting ERCOT ML Inference Server
========================================

Model: lgbm
Port: 5001

========================================
Container Started Successfully!
========================================

Health: http://localhost:5001/health
Score:  http://localhost:5001/score
Info:   http://localhost:5001/model/info
```

**The server is now running!** 🎉

---

### 3. Test the Endpoints

**In Command Prompt:**
```cmd
test.bat
```

**In PowerShell:**
```powershell
.\test.bat
```

**In Git Bash:**
```bash
bash test.sh
```

**Expected output:**
```
========================================
Testing ERCOT ML Pipeline
========================================

[1/4] Testing Health Endpoint
========================================
{"status":"healthy","model_type":"lgbm","model_loaded":true,"timestamp":"2024-07-01T12:00:00"}

[2/4] Testing Model Info Endpoint
========================================
{"model_type":"lgbm","feature_count":50,...}

[3/4] Testing Scoring Endpoint
========================================
{"predictions":[3.25,3.75],"model_type":"lgbm","timestamp":"2024-07-01T12:00:00","count":2}

========================================
Tests Complete!
========================================
```

---

## 🌐 Access the API

Once running, you can access the API at:

- **Health Check**: http://localhost:5001/health
- **API Documentation**: http://localhost:5001/docs (Swagger UI)
- **Score Endpoint**: http://localhost:5001/score

### Test with Browser

Open http://localhost:5001/docs in your browser to see the **interactive API documentation**!

### Test with curl

```cmd
curl http://localhost:5001/health
```

### Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:5001/health")
print(response.json())

# Make prediction
payload = {
    "data": [{
        "TimestampHour": "2024-07-01 15:00:00",
        "SettlementPoint": "HB_HOUSTON",
        "Load_NORTH_Hourly": 50000,
        "DAM_Price_Hourly": 25.75,
        "RTM_LMP_HourlyAvg": 22.50
    }]
}
response = requests.post("http://localhost:5001/score", json=payload)
print(response.json())
```

---

## 🛠️ Common Tasks

### View Container Logs
```cmd
logs.bat
```

Or:
```cmd
docker logs -f ercot-ml-pipeline
```

### Stop the Container
```cmd
stop.bat
```

Or:
```cmd
docker stop ercot-ml-pipeline
docker rm ercot-ml-pipeline
```

### Restart with Different Model
```cmd
# Stop first
stop.bat

# Edit run.bat and change MODEL_TYPE to xgb or deep
notepad run.bat

# Start again
run.bat
```

### Check if Container is Running
```cmd
docker ps
```

### Shell into Container
```cmd
docker exec -it ercot-ml-pipeline /bin/bash
```

---

## 🐛 Troubleshooting

### Issue: "docker: command not found"

**Solution**: Docker Desktop is not installed or not running.

1. Download from: https://www.docker.com/products/docker-desktop/
2. Install and restart Windows
3. Ensure Docker Desktop is running (check system tray)

### Issue: "Error response from daemon: Conflict"

**Solution**: Container with same name already exists.

```cmd
docker stop ercot-ml-pipeline
docker rm ercot-ml-pipeline
run.bat
```

### Issue: "Cannot connect to the Docker daemon"

**Solution**: Docker Desktop is not running.

1. Start Docker Desktop from Start Menu
2. Wait for it to fully start (whale icon in system tray)
3. Try again

### Issue: Build fails with "No space left on device"

**Solution**: Docker needs more disk space.

```cmd
# Clean up unused Docker resources
docker system prune -a
```

### Issue: Port 5001 already in use

**Solution**: Change the port in `run.bat`:

```cmd
set PORT=5002
```

---

## 📊 Windows-Specific Notes

### File Paths

Windows uses backslashes, but Docker prefers forward slashes. The batch files handle this automatically:

```cmd
# Correct (batch file does this)
-v "%CD%\models:/app/models"

# Also works in PowerShell
-v "${PWD}/models:/app/models"
```

### Line Endings

If you edit files on Windows, they may have CRLF line endings. Docker handles this automatically, but if you see errors about `\r`, convert to LF:

```cmd
# In Git Bash
dos2unix filename.sh
```

### Performance

Docker Desktop on Windows uses WSL2 for better performance. Make sure WSL2 is enabled:

```cmd
wsl --set-default-version 2
```

---

## 🎯 Complete Windows Workflow

```cmd
# 1. Build (first time only)
build.bat

# 2. Start server
run.bat

# 3. Test it works
test.bat

# 4. Use the API
curl http://localhost:5001/health

# 5. View logs (optional)
logs.bat

# 6. Stop when done
stop.bat
```

---

## 💡 Pro Tips for Windows Users

### 1. Use Windows Terminal

Download from Microsoft Store for a better terminal experience with tabs:
- Supports Command Prompt, PowerShell, and Git Bash in one app
- Better colors and fonts

### 2. Install WSL2

For better Docker performance:
```cmd
wsl --install
```

### 3. Use Visual Studio Code

Built-in Docker extension for Windows:
- Right-click Dockerfile → Build Image
- View containers in sidebar
- Attach shell to running containers

### 4. Create Desktop Shortcuts

Create shortcuts to the batch files:
- Right-click `run.bat` → Send to → Desktop (create shortcut)
- Double-click to start the server instantly

---

## ✅ Verification Checklist

- [ ] Docker Desktop installed and running
- [ ] `build.bat` completes successfully
- [ ] `run.bat` starts container
- [ ] http://localhost:5001/health returns 200
- [ ] `test.bat` passes all tests
- [ ] Can view logs with `logs.bat`
- [ ] Can stop with `stop.bat`

---

## 📚 Additional Resources

- **Docker Desktop Docs**: https://docs.docker.com/desktop/windows/
- **WSL2 Setup**: https://docs.microsoft.com/en-us/windows/wsl/install
- **Windows Terminal**: https://aka.ms/terminal

---

## 🎊 You're All Set!

Your ERCOT ML pipeline is now running on Windows!

**Next steps:**
1. Run `build.bat` to build the container
2. Run `run.bat` to start the server
3. Open http://localhost:5001/docs in your browser

**Need help?** Check the main documentation:
- `STEP5_CONTAINERIZATION.md` - Complete container guide
- `DEPLOYMENT_GUIDE.md` - Production deployment
- `QUICK_START.md` - General quick start

---

**Windows commands work exactly like the Makefile commands - just use `.bat` files instead!** 🪟✅

