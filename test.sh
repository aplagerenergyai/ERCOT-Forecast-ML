#!/bin/bash
# ============================================================================
# test.sh - Test inference endpoints on Windows (Git Bash)
# ============================================================================

PORT=5001
CONTAINER_NAME="ercot-ml-pipeline"

echo ""
echo "========================================"
echo "Testing ERCOT ML Pipeline"
echo "========================================"
echo ""

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Container not running. Starting..."
    bash run.sh
    sleep 10
fi

echo ""
echo "[1/4] Testing Health Endpoint"
echo "========================================"
curl -s http://localhost:$PORT/health | python -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/health
echo ""

echo ""
echo "[2/4] Testing Model Info Endpoint"
echo "========================================"
curl -s http://localhost:$PORT/model/info | python -m json.tool 2>/dev/null | head -n 20 || curl -s http://localhost:$PORT/model/info
echo ""

echo ""
echo "[3/4] Testing Scoring Endpoint"
echo "========================================"
curl -s -X POST http://localhost:$PORT/score \
    -H "Content-Type: application/json" \
    -d @local_test_payload.json | python -m json.tool 2>/dev/null || curl -s -X POST http://localhost:$PORT/score \
    -H "Content-Type: application/json" \
    -d @local_test_payload.json
echo ""

echo ""
echo "========================================"
echo "Tests Complete!"
echo "========================================"
echo ""

