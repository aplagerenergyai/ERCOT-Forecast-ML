#!/bin/bash
# ============================================================================
# logs.sh - View container logs on Windows (Git Bash)
# ============================================================================

CONTAINER_NAME="ercot-ml-pipeline"

echo ""
echo "Container logs (press Ctrl+C to exit):"
echo ""

docker logs -f $CONTAINER_NAME

