#!/usr/bin/env bash
# Check SSH tunnel status (simple version)

TUNNEL_SESSION="ssh-tunnel"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}SSH Tunnel Status${NC}"
echo "======================================"
echo ""

# Check tmux session
if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${GREEN}✓ Tmux session 'ssh-tunnel' is RUNNING${NC}"
    
    # Check ports
    echo ""
    echo "Port status:"
    if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/8000" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Retriever (127.0.0.1:8000) - OK"
    else
        echo -e "  ${RED}✗${NC} Retriever (127.0.0.1:8000) - NOT RESPONDING"
    fi
    
    if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/8127" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Analyzer (127.0.0.1:8127) - OK"
    else
        echo -e "  ${RED}✗${NC} Analyzer (127.0.0.1:8127) - NOT RESPONDING"
    fi
    
    echo ""
    echo -e "${YELLOW}Tunnel session output (last 5 lines):${NC}"
    tmux capture-pane -t "$TUNNEL_SESSION" -p | tail -5
    echo ""
    echo -e "${YELLOW}To attach to tunnel:${NC}"
    echo "  tmux attach -t $TUNNEL_SESSION"
else
    echo -e "${RED}✗ Tmux session 'ssh-tunnel' is NOT RUNNING${NC}"
    echo ""
    echo "To start the tunnel:"
    echo -e "  ${YELLOW}bash /home/luwa/Documents/Tree-GRPO/start_tunnel_simple.sh${NC}"
fi
