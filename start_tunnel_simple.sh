#!/usr/bin/env bash
# Start SSH tunnel in tmux (simple version, no dependencies)

TUNNEL_SESSION="ssh-tunnel"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}SSH Tunnel Manager (Simple)${NC}"
echo "======================================"
echo ""

# Check if session already running
if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${YELLOW}Tunnel session '$TUNNEL_SESSION' already running!${NC}"
    echo ""
    echo "To attach and see the tunnel:"
    echo -e "  ${YELLOW}tmux attach -t $TUNNEL_SESSION${NC}"
    echo ""
    echo "To stop:"
    echo -e "  ${YELLOW}tmux kill-session -t $TUNNEL_SESSION${NC}"
    exit 0
fi

# Create tmux session with SSH tunnel
echo -e "${YELLOW}Creating SSH tunnel in tmux session '$TUNNEL_SESSION'...${NC}"
echo -e "${YELLOW}You will be prompted to enter SSH password.${NC}"
echo ""

tmux new-session -d -s "$TUNNEL_SESSION" \
    "ssh -L 8000:127.0.0.1:8000 -L 8127:127.0.0.1:8127 -N -o ServerAliveInterval=60 luwa@172.18.167.248"

sleep 2

if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH tunnel created!${NC}"
    echo ""
    echo -e "${YELLOW}Session Details:${NC}"
    echo "  Session name: $TUNNEL_SESSION"
    echo "  Tunnel: localhost:8000 ←→ 172.18.167.248:8000 (Retriever)"
    echo "  Tunnel: localhost:8127 ←→ 172.18.167.248:8127 (Analyzer)"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  View tunnel: tmux attach -t $TUNNEL_SESSION"
    echo "  Stop tunnel: tmux kill-session -t $TUNNEL_SESSION"
    echo "  Check status: bash /home/luwa/Documents/Tree-GRPO/check_tunnel.sh"
    echo ""
    echo -e "${YELLOW}Note:${NC}"
    echo "  - The tunnel will continue running even after you close this terminal"
    echo "  - Password: Enter your SSH password when prompted (invisible)"
    echo "  - Connection will auto-restart if interrupted"
else
    echo -e "${RED}✗ Failed to create tunnel${NC}"
    exit 1
fi
