#!/bin/bash
# ============================================================================
# build.sh - Build Docker image on Windows (Git Bash)
# ============================================================================

echo ""
echo "========================================"
echo "Building ERCOT ML Pipeline Docker Image"
echo "========================================"
echo ""

docker build -t ercot-ml-pipeline:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build Complete!"
    echo "========================================"
    echo ""
    echo "Image: ercot-ml-pipeline:latest"
    echo "Next: bash run.sh to start the container"
else
    echo ""
    echo "========================================"
    echo "Build Failed!"
    echo "========================================"
    exit 1
fi

