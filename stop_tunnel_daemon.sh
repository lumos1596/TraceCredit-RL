#!/usr/bin/env bash
# Stop SSH tunnel daemon

TUNNEL_SESSION="ssh-tunnel-service"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}tmux not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Stopping SSH tunnel daemon...${NC}"

if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    tmux kill-session -t "$TUNNEL_SESSION"
    sleep 1
    
    # Verify termination
    if ! tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH tunnel daemon stopped${NC}"
        
        # Clean up PID file
        rm -f /tmp/ssh_tunnel_172.18.167.248.pid
        
        echo ""
        echo "To view the tunnel logs:"
        echo -e "  ${YELLOW}tail -f /tmp/ssh_tunnel_172.18.167.248.log${NC}"
    else
        echo -e "${RED}✗ Failed to stop tunnel session${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}No active tunnel session found${NC}"
fi
