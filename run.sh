#!/bin/bash
# ============================================================================
# run.sh - Run Docker container on Windows (Git Bash)
# ============================================================================

IMAGE_NAME="ercot-ml-pipeline"
CONTAINER_NAME="ercot-ml-pipeline"
PORT=5001
MODEL_TYPE="lgbm"

echo ""
echo "========================================"
echo "Starting ERCOT ML Inference Server"
echo "========================================"
echo ""
echo "Model: $MODEL_TYPE"
echo "Port: $PORT"
echo ""

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Start new container
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:5001 \
    -e MODEL_TYPE=$MODEL_TYPE \
    -e LOG_LEVEL=INFO \
    -v "$(pwd)/models:/app/models:ro" \
    $IMAGE_NAME:latest

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Container Started Successfully!"
    echo "========================================"
    echo ""
    echo "Health: http://localhost:$PORT/health"
    echo "Score:  http://localhost:$PORT/score"
    echo "Info:   http://localhost:$PORT/model/info"
    echo ""
    echo "Logs:   docker logs $CONTAINER_NAME"
    echo "Stop:   docker stop $CONTAINER_NAME"
    echo ""
    sleep 5
    echo "Testing health endpoint..."
    curl -s http://localhost:$PORT/health | python -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/health
else
    echo ""
    echo "========================================"
    echo "Failed to Start Container!"
    echo "========================================"
    exit 1
fi

