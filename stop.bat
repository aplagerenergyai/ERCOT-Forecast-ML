@echo off
REM ============================================================================
REM stop.bat - Stop Docker container on Windows
REM ============================================================================

set CONTAINER_NAME=ercot-ml-pipeline

echo.
echo Stopping container: %CONTAINER_NAME%
echo.

docker stop %CONTAINER_NAME%
docker rm %CONTAINER_NAME%

echo.
echo Container stopped and removed.
echo.

