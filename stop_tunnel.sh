#!/usr/bin/env bash
# Stop SSH tunnel

TUNNEL_SESSION="ssh-tunnel"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}tmux not found${NC}"
    exit 1
fi

if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${YELLOW}Stopping SSH tunnel...${NC}"
    tmux kill-session -t "$TUNNEL_SESSION"
    sleep 1
    
    if ! tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
        echo -e "${GREEN}✓ SSH tunnel stopped${NC}"
    else
        echo -e "${RED}✗ Failed to stop tunnel${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Tunnel session not running${NC}"
fi
