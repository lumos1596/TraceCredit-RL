#!/usr/bin/env bash
# SSH tunnel setup for accessing remote services
# This script establishes SSH tunnels to access retriever and analyzer services
# running on the service host (172.18.167.248)

set -euo pipefail

SERVICE_HOST="172.18.167.248"
SERVICE_USER="${SERVICE_USER:-luwa}"

# Local ports (on this machine)
LOCAL_RETRIEVER_PORT=8000
LOCAL_ANALYZER_PORT=8127

# Remote ports (on service host)
REMOTE_RETRIEVER_PORT=8000
REMOTE_ANALYZER_PORT=8127

# SSH tunnel process file
TUNNEL_PID_FILE="/tmp/ssh_tunnel_${SERVICE_HOST}.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up SSH tunnel to ${SERVICE_HOST}...${NC}"

# Check if tunnel already exists
if [[ -f "$TUNNEL_PID_FILE" ]]; then
    OLD_PID=$(cat "$TUNNEL_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH tunnel already running (PID: $OLD_PID)${NC}"
        exit 0
    else
        echo -e "${YELLOW}Old tunnel PID not running, cleaning up...${NC}"
        rm -f "$TUNNEL_PID_FILE"
    fi
fi

# Establish SSH tunnel in background
# -L local_port:remote_host:remote_port
# -N don't execute remote command
# -f fork to background
echo "Establishing tunnels:"
echo "  - Retriever: localhost:${LOCAL_RETRIEVER_PORT} → ${SERVICE_HOST}:${REMOTE_RETRIEVER_PORT}"
echo "  - Analyzer:  localhost:${LOCAL_ANALYZER_PORT} → ${SERVICE_HOST}:${REMOTE_ANALYZER_PORT}"
echo ""
echo "Please enter SSH password for ${SERVICE_USER}@${SERVICE_HOST}:"

ssh -L ${LOCAL_RETRIEVER_PORT}:127.0.0.1:${REMOTE_RETRIEVER_PORT} \
    -L ${LOCAL_ANALYZER_PORT}:127.0.0.1:${REMOTE_ANALYZER_PORT} \
    -N -f "${SERVICE_USER}@${SERVICE_HOST}"

# For background process, we can't reliably get PID, so we find it by ssh command
sleep 1
TUNNEL_PID=$(pgrep -f "ssh.*${LOCAL_RETRIEVER_PORT}.*${LOCAL_ANALYZER_PORT}.*${SERVICE_HOST}" | head -1)

if [[ -z "$TUNNEL_PID" ]]; then
    echo -e "${RED}✗ Failed to establish SSH tunnel${NC}"
    exit 1
fi

echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"

# Give the tunnel time to establish
sleep 2

# Verify tunnel is working
echo ""
echo "Verifying tunnel connections..."

# Test retriever
if timeout 3 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_RETRIEVER_PORT}" 2>/dev/null; then
    echo -e "${GREEN}✓ Retriever tunnel OK (127.0.0.1:${LOCAL_RETRIEVER_PORT})${NC}"
else
    echo -e "${RED}✗ Retriever tunnel FAILED${NC}"
    kill $TUNNEL_PID 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
    exit 1
fi

# Test analyzer
if timeout 3 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_ANALYZER_PORT}" 2>/dev/null; then
    echo -e "${GREEN}✓ Analyzer tunnel OK (127.0.0.1:${LOCAL_ANALYZER_PORT})${NC}"
else
    echo -e "${RED}✗ Analyzer tunnel FAILED${NC}"
    kill $TUNNEL_PID 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All tunnels established successfully!${NC}"
echo ""
echo "Tunnel information:"
echo "  PID: $TUNNEL_PID"
echo "  Config file: $TUNNEL_PID_FILE"
echo ""
echo "To stop the tunnel, run:"
echo "  kill $TUNNEL_PID"
echo "  # or"
echo "  bash stop_ssh_tunnel.sh"
echo ""
echo "Environment variables are already set to use tunneled services:"
echo "  RETRIEVER_URL=http://127.0.0.1:8000/retrieve"
echo "  SELF_OPD_ANALYZER_URL=http://127.0.0.1:8127"
