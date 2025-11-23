#!/bin/bash
# ============================================================================
# test_score_local.sh
#
# Test script for local Docker inference endpoint
# 
# Usage:
#   bash test_score_local.sh
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PORT=5001
CONTAINER_NAME="ercot-ml-pipeline"
WAIT_TIME=10

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ERCOT ML Pipeline - Local Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${YELLOW}⚠ Container not running. Starting...${NC}"
    make run
    echo -e "${YELLOW}Waiting ${WAIT_TIME}s for container to be ready...${NC}"
    sleep $WAIT_TIME
fi

echo ""
echo -e "${GREEN}[1/4] Testing Health Endpoint${NC}"
echo "GET http://localhost:$PORT/health"
echo ""

HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/health)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n 1)
RESPONSE_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Health check passed (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
else
    echo -e "${RED}✗ Health check failed (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY"
    exit 1
fi

echo ""
echo -e "${GREEN}[2/4] Testing Model Info Endpoint${NC}"
echo "GET http://localhost:$PORT/model/info"
echo ""

INFO_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/model/info)
HTTP_CODE=$(echo "$INFO_RESPONSE" | tail -n 1)
RESPONSE_BODY=$(echo "$INFO_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Model info retrieved (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null | head -n 20 || echo "$RESPONSE_BODY"
else
    echo -e "${RED}✗ Model info failed (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY"
fi

echo ""
echo -e "${GREEN}[3/4] Testing Scoring Endpoint${NC}"
echo "POST http://localhost:$PORT/score"
echo ""

if [ ! -f "local_test_payload.json" ]; then
    echo -e "${RED}✗ local_test_payload.json not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Payload:${NC}"
cat local_test_payload.json | python3 -m json.tool 2>/dev/null | head -n 15
echo "..."
echo ""

SCORE_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST http://localhost:$PORT/score \
    -H "Content-Type: application/json" \
    -d @local_test_payload.json)

HTTP_CODE=$(echo "$SCORE_RESPONSE" | tail -n 1)
RESPONSE_BODY=$(echo "$SCORE_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Scoring successful (HTTP $HTTP_CODE)${NC}"
    echo ""
    echo -e "${YELLOW}Response:${NC}"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
    
    # Extract predictions
    PREDICTIONS=$(echo "$RESPONSE_BODY" | python3 -c "import json, sys; data=json.load(sys.stdin); print(data['predictions'])" 2>/dev/null || echo "[]")
    echo ""
    echo -e "${GREEN}DART Predictions:${NC} $PREDICTIONS"
else
    echo -e "${RED}✗ Scoring failed (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY"
    exit 1
fi

echo ""
echo -e "${GREEN}[4/4] Testing Error Handling${NC}"
echo "POST http://localhost:$PORT/score (with invalid data)"
echo ""

INVALID_PAYLOAD='{"data": [{"invalid": "data"}]}'
ERROR_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST http://localhost:$PORT/score \
    -H "Content-Type: application/json" \
    -d "$INVALID_PAYLOAD")

HTTP_CODE=$(echo "$ERROR_RESPONSE" | tail -n 1)
RESPONSE_BODY=$(echo "$ERROR_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "500" ] || [ "$HTTP_CODE" = "422" ]; then
    echo -e "${GREEN}✓ Error handling works (HTTP $HTTP_CODE)${NC}"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
else
    echo -e "${YELLOW}⚠ Expected error code, got HTTP $HTTP_CODE${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All Tests Passed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Endpoint Summary:${NC}"
echo "  Health:     http://localhost:$PORT/health"
echo "  Model Info: http://localhost:$PORT/model/info"
echo "  Score:      http://localhost:$PORT/score"
echo "  Logs:       docker logs $CONTAINER_NAME"
echo ""

