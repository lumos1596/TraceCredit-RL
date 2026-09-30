#!/usr/bin/env bash
# Long-running SSH tunnel using tmux
# This keeps the tunnel alive by restarting if needed

set -euo pipefail

SERVICE_HOST="172.18.167.248"
SERVICE_USER="${SERVICE_USER:-luwa}"

LOCAL_RETRIEVER_PORT=8000
LOCAL_ANALYZER_PORT=8127
REMOTE_RETRIEVER_PORT=8000
REMOTE_ANALYZER_PORT=8127

TUNNEL_LOG="/tmp/ssh_tunnel_daemon.log"
MONITOR_INTERVAL=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $@"
    echo -e "${BLUE}$msg${NC}" | tee -a "$TUNNEL_LOG"
}

success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $@"
    echo -e "${GREEN}$msg${NC}" | tee -a "$TUNNEL_LOG"
}

error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $@"
    echo -e "${RED}$msg${NC}" | tee -a "$TUNNEL_LOG"
}

# Initialize log
{
    echo "==================== SSH Tunnel Daemon ===================="
    echo "Started: $(date)"
    echo "Service Host: $SERVICE_HOST"
    echo "==========================================================="
} >> "$TUNNEL_LOG"

log "SSH Tunnel Daemon starting..."
log "This script will establish and monitor the SSH tunnel."
log "Password will be requested once."

# Function to check if tunnel is alive
check_tunnel() {
    if ! timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_RETRIEVER_PORT}" 2>/dev/null; then
        return 1
    fi
    if ! timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_ANALYZER_PORT}" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Function to establish tunnel
establish_tunnel() {
    log "Attempting to establish SSH tunnel..."
    
    # Kill any existing tunnels on these ports
    pkill -f "ssh.*${LOCAL_RETRIEVER_PORT}.*${LOCAL_ANALYZER_PORT}" || true
    sleep 1
    
    # Start new tunnel - this will prompt for password interactively
    ssh -L ${LOCAL_RETRIEVER_PORT}:127.0.0.1:${REMOTE_RETRIEVER_PORT} \
        -L ${LOCAL_ANALYZER_PORT}:127.0.0.1:${REMOTE_ANALYZER_PORT} \
        -N \
        -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        "${SERVICE_USER}@${SERVICE_HOST}"
}

# Main loop
success "SSH Tunnel Daemon ready."
success "The tunnel will stay open. To close it, press Ctrl+C or close this terminal."
log "To keep this running in the background, use:"
log "  tmux new-session -d -s ssh-tunnel 'bash $0'"
log "Or:"
log "  nohup bash $0 > /tmp/tunnel_daemon.log 2>&1 &"
echo ""
log "Establishing initial tunnel connection..."
log "You will be prompted to enter SSH password for ${SERVICE_USER}@${SERVICE_HOST}"
echo ""

# Attempt to establish tunnel
establish_tunnel

# This is the main loop - the SSH tunnel runs in foreground above
# If SSH tunnel dies, it will exit and we detect it
log "Tunnel connection closed or interrupted."
