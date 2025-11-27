@echo off
REM ============================================================================
REM build.bat - Build Docker image on Windows
REM ============================================================================

echo.
echo ========================================
echo Building ERCOT ML Pipeline Docker Image
echo ========================================
echo.

docker build -t ercot-ml-pipeline:latest .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build Complete!
    echo ========================================
    echo.
    echo Image: ercot-ml-pipeline:latest
    echo Next: run.bat to start the container
) else (
    echo.
    echo ========================================
    echo Build Failed!
    echo ========================================
    exit /b 1
)

