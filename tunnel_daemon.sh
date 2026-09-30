#!/usr/bin/env bash
# Long-running SSH tunnel daemon
# This script keeps the SSH tunnel alive and auto-restarts if it dies

set -euo pipefail

SERVICE_HOST="172.18.167.248"
SERVICE_USER="${SERVICE_USER:-luwa}"
SERVICE_PASSWORD="${SERVICE_PASSWORD:-20030529Llw!}"

LOCAL_RETRIEVER_PORT=8000
LOCAL_ANALYZER_PORT=8127
REMOTE_RETRIEVER_PORT=8000
REMOTE_ANALYZER_PORT=8127

TUNNEL_PID_FILE="/tmp/ssh_tunnel_${SERVICE_HOST}.pid"
TUNNEL_LOG_FILE="/tmp/ssh_tunnel_${SERVICE_HOST}.log"
MONITOR_INTERVAL=30  # Check tunnel every 30 seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $@" | tee -a "$TUNNEL_LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $@" | tee -a "$TUNNEL_LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $@" | tee -a "$TUNNEL_LOG_FILE"
}

# Initialize log file
{
    echo "==================== SSH Tunnel Daemon Log ===================="
    echo "Started: $(date)"
    echo "Service Host: $SERVICE_HOST"
    echo "Tunnel: localhost:$LOCAL_RETRIEVER_PORT → $SERVICE_HOST:$REMOTE_RETRIEVER_PORT"
    echo "Tunnel: localhost:$LOCAL_ANALYZER_PORT → $SERVICE_HOST:$REMOTE_ANALYZER_PORT"
    echo "=============================================================="
} >> "$TUNNEL_LOG_FILE"

log "SSH Tunnel Daemon starting..."

# Check if sshpass is available
if ! command -v sshpass &> /dev/null; then
    error "sshpass not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y sshpass
    elif command -v yum &> /dev/null; then
        sudo yum install -y sshpass
    elif command -v brew &> /dev/null; then
        brew install sshpass
    else
        error "Cannot install sshpass. Please install manually."
        exit 1
    fi
fi

establish_tunnel() {
    log "Establishing SSH tunnel..."
    
    # Kill any existing tunnels
    if [[ -f "$TUNNEL_PID_FILE" ]]; then
        OLD_PID=$(cat "$TUNNEL_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            log "Killing old tunnel (PID: $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 1
            if kill -0 "$OLD_PID" 2>/dev/null; then
                kill -9 "$OLD_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Establish new tunnel
    export SSHPASS="$SERVICE_PASSWORD"
    sshpass -e ssh \
        -L ${LOCAL_RETRIEVER_PORT}:127.0.0.1:${REMOTE_RETRIEVER_PORT} \
        -L ${LOCAL_ANALYZER_PORT}:127.0.0.1:${REMOTE_ANALYZER_PORT} \
        -N -f \
        -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile=/dev/null \
        "${SERVICE_USER}@${SERVICE_HOST}"
    
    sleep 2
    TUNNEL_PID=$(pgrep -f "ssh.*${LOCAL_RETRIEVER_PORT}.*${LOCAL_ANALYZER_PORT}.*${SERVICE_HOST}" | head -1)
    
    if [[ -z "$TUNNEL_PID" ]]; then
        error "Failed to establish SSH tunnel"
        return 1
    fi
    
    echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"
    success "SSH tunnel established (PID: $TUNNEL_PID)"
    return 0
}

verify_tunnel() {
    if [[ ! -f "$TUNNEL_PID_FILE" ]]; then
        return 1
    fi
    
    PID=$(cat "$TUNNEL_PID_FILE")
    if ! kill -0 "$PID" 2>/dev/null; then
        return 1
    fi
    
    # Check if ports are actually listening
    if ! timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_RETRIEVER_PORT}" 2>/dev/null; then
        return 1
    fi
    
    if ! timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${LOCAL_ANALYZER_PORT}" 2>/dev/null; then
        return 1
    fi
    
    return 0
}

# Signal handlers
cleanup() {
    log "Shutting down SSH Tunnel Daemon..."
    if [[ -f "$TUNNEL_PID_FILE" ]]; then
        PID=$(cat "$TUNNEL_PID_FILE")
        kill "$PID" 2>/dev/null || true
        rm -f "$TUNNEL_PID_FILE"
    fi
    log "SSH Tunnel Daemon stopped."
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main daemon loop
establish_tunnel

while true; do
    if ! verify_tunnel; then
        error "Tunnel is not responding, attempting to restart..."
        establish_tunnel
    else
        log "Tunnel is running (PID: $(cat "$TUNNEL_PID_FILE"))"
    fi
    
    sleep $MONITOR_INTERVAL
done
