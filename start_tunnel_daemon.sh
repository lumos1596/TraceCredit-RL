#!/usr/bin/env bash
# Start SSH tunnel in a tmux session (long-running)

set -euo pipefail

TUNNEL_SESSION="ssh-tunnel-service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$SCRIPT_DIR/tunnel_daemon.sh"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}SSH Tunnel Manager${NC}"
echo "This script will start the SSH tunnel in a tmux session for long-running operation."
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}tmux not found. Installing...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y tmux
    elif command -v yum &> /dev/null; then
        sudo yum install -y tmux
    elif command -v brew &> /dev/null; then
        brew install tmux
    else
        echo "Cannot install tmux. Please install manually."
        exit 1
    fi
fi

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo -e "${YELLOW}sshpass not found. Installing...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y sshpass
    elif command -v yum &> /dev/null; then
        sudo yum install -y sshpass
    elif command -v brew &> /dev/null; then
        brew install sshpass
    else
        echo "Cannot install sshpass. Please install manually."
        exit 1
    fi
fi

# Check if session already exists
if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${YELLOW}Tmux session '$TUNNEL_SESSION' already exists.${NC}"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $TUNNEL_SESSION"
    echo "  2. Kill and restart: tmux kill-session -t $TUNNEL_SESSION && bash start_tunnel_daemon.sh"
    echo "  3. View status: tmux capture-pane -t $TUNNEL_SESSION -p"
    exit 0
fi

# Create new tmux session with the daemon script
echo -e "${YELLOW}Creating tmux session '$TUNNEL_SESSION'...${NC}"
chmod +x "$DAEMON_SCRIPT"
tmux new-session -d -s "$TUNNEL_SESSION" "bash $DAEMON_SCRIPT"

sleep 2

# Check if session started successfully
if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH tunnel started in tmux session!${NC}"
    echo ""
    echo "Session name: $TUNNEL_SESSION"
    echo ""
    echo "To attach to the tunnel session:"
    echo -e "  ${YELLOW}tmux attach -t $TUNNEL_SESSION${NC}"
    echo ""
    echo "To view logs:"
    echo -e "  ${YELLOW}tail -f /tmp/ssh_tunnel_172.18.167.248.log${NC}"
    echo ""
    echo "To stop the tunnel:"
    echo -e "  ${YELLOW}tmux kill-session -t $TUNNEL_SESSION${NC}"
    echo ""
    echo "To restart the tunnel:"
    echo -e "  ${YELLOW}tmux kill-session -t $TUNNEL_SESSION && bash start_tunnel_daemon.sh${NC}"
else
    echo -e "${RED}✗ Failed to start SSH tunnel${NC}"
    exit 1
fi
