@echo off
REM ============================================================================
REM test.bat - Test inference endpoints on Windows
REM ============================================================================

set PORT=5001
set CONTAINER_NAME=ercot-ml-pipeline

echo.
echo ========================================
echo Testing ERCOT ML Pipeline
echo ========================================
echo.

REM Check if container is running
docker ps | findstr %CONTAINER_NAME% >nul
if %ERRORLEVEL% NEQ 0 (
    echo Container not running. Starting...
    call run.bat
    timeout /t 10 /nobreak >nul
)

echo.
echo [1/4] Testing Health Endpoint
echo ========================================
curl -s http://localhost:%PORT%/health
echo.

echo.
echo [2/4] Testing Model Info Endpoint
echo ========================================
curl -s http://localhost:%PORT%/model/info
echo.

echo.
echo [3/4] Testing Scoring Endpoint
echo ========================================
curl -s -X POST http://localhost:%PORT%/score ^
    -H "Content-Type: application/json" ^
    -d @local_test_payload.json
echo.

echo.
echo ========================================
echo Tests Complete!
echo ========================================
echo.

