#!/bin/bash
# ============================================================================
# stop.sh - Stop Docker container on Windows (Git Bash)
# ============================================================================

CONTAINER_NAME="ercot-ml-pipeline"

echo ""
echo "Stopping container: $CONTAINER_NAME"
echo ""

docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo ""
echo "Container stopped and removed."
echo ""

