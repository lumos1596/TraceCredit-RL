#!/usr/bin/env bash
# Check SSH tunnel daemon status

TUNNEL_SESSION="ssh-tunnel-service"
TUNNEL_PID_FILE="/tmp/ssh_tunnel_172.18.167.248.pid"
TUNNEL_LOG_FILE="/tmp/ssh_tunnel_172.18.167.248.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}SSH Tunnel Daemon Status${NC}"
echo "======================================"

# Check tmux session
if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${GREEN}✓ Tmux session '$TUNNEL_SESSION' is running${NC}"
else
    echo -e "${RED}✗ Tmux session '$TUNNEL_SESSION' is not running${NC}"
    exit 1
fi

# Check PID file
if [[ -f "$TUNNEL_PID_FILE" ]]; then
    PID=$(cat "$TUNNEL_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH tunnel process (PID: $PID) is running${NC}"
    else
        echo -e "${RED}✗ SSH tunnel process (PID: $PID) is not running${NC}"
    fi
else
    echo -e "${YELLOW}⚠ PID file not found${NC}"
fi

# Check port connectivity
echo ""
echo "Port connectivity:"
if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/8000" 2>/dev/null; then
    echo -e "${GREEN}✓ Retriever port 8000 is accessible${NC}"
else
    echo -e "${RED}✗ Retriever port 8000 is not accessible${NC}"
fi

if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/8127" 2>/dev/null; then
    echo -e "${GREEN}✓ Analyzer port 8127 is accessible${NC}"
else
    echo -e "${RED}✗ Analyzer port 8127 is not accessible${NC}"
fi

# Show recent logs
echo ""
echo "Recent logs:"
if [[ -f "$TUNNEL_LOG_FILE" ]]; then
    echo -e "${YELLOW}(Last 10 lines)${NC}"
    tail -10 "$TUNNEL_LOG_FILE"
else
    echo "No log file found"
fi

# Show tmux session output
echo ""
echo "Tmux session output:"
echo -e "${YELLOW}(Last 5 lines)${NC}"
tmux capture-pane -t "$TUNNEL_SESSION" -p | tail -5
