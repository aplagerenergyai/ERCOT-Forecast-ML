@echo off
REM ============================================================================
REM run.bat - Run Docker container on Windows
REM ============================================================================

set IMAGE_NAME=ercot-ml-pipeline
set CONTAINER_NAME=ercot-ml-pipeline
set PORT=5001
set MODEL_TYPE=lgbm

echo.
echo ========================================
echo Starting ERCOT ML Inference Server
echo ========================================
echo.
echo Model: %MODEL_TYPE%
echo Port: %PORT%
echo.

REM Stop existing container if running
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul

REM Start new container
docker run -d ^
    --name %CONTAINER_NAME% ^
    -p %PORT%:5001 ^
    -e MODEL_TYPE=%MODEL_TYPE% ^
    -e LOG_LEVEL=INFO ^
    -v "%CD%\models:/app/models:ro" ^
    %IMAGE_NAME%:latest

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Container Started Successfully!
    echo ========================================
    echo.
    echo Health: http://localhost:%PORT%/health
    echo Score:  http://localhost:%PORT%/score
    echo Info:   http://localhost:%PORT%/model/info
    echo.
    echo Logs:   docker logs %CONTAINER_NAME%
    echo Stop:   docker stop %CONTAINER_NAME%
    echo.
    timeout /t 5 /nobreak >nul
    echo Testing health endpoint...
    curl -s http://localhost:%PORT%/health
) else (
    echo.
    echo ========================================
    echo Failed to Start Container!
    echo ========================================
    exit /b 1
)

