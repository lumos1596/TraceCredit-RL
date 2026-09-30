#!/usr/bin/env bash
# Stop SSH tunnel

TUNNEL_PID_FILE="/tmp/ssh_tunnel_172.18.167.248.pid"

if [[ -f "$TUNNEL_PID_FILE" ]]; then
    PID=$(cat "$TUNNEL_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Killing SSH tunnel (PID: $PID)..."
        kill "$PID"
        sleep 1
        if kill -0 "$PID" 2>/dev/null; then
            echo "Forcefully killing SSH tunnel..."
            kill -9 "$PID"
        fi
        rm -f "$TUNNEL_PID_FILE"
        echo "SSH tunnel stopped."
    else
        echo "SSH tunnel PID not running. Cleaning up PID file..."
        rm -f "$TUNNEL_PID_FILE"
    fi
else
    echo "No SSH tunnel PID file found. Tunnel may not be running."
    # Try to kill any SSH process forwarding to our ports
    pkill -f "ssh.*8000.*8127" || echo "No matching SSH processes found."
fi
