@echo off
REM ============================================================================
REM logs.bat - View container logs on Windows
REM ============================================================================

set CONTAINER_NAME=ercot-ml-pipeline

echo.
echo Container logs (press Ctrl+C to exit):
echo.

docker logs -f %CONTAINER_NAME%

