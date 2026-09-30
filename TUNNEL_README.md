# SSH Tunnel Setup & Management

This directory contains scripts to manage the SSH tunnel for accessing remote services running on `172.18.167.248`.

## Quick Start

### Start the tunnel (long-running, in background)
```bash
bash start_tunnel_simple.sh
```

This will:
1. Create a tmux session called `ssh-tunnel`
2. Establish SSH tunnels:
   - Local `127.0.0.1:8000` → Remote `172.18.167.248:8000` (Retriever)
   - Local `127.0.0.1:8127` → Remote `172.18.167.248:8127` (Analyzer)
3. The tunnel stays alive even after closing the terminal

### Check tunnel status
```bash
bash check_tunnel.sh
```

Shows:
- ✓/✗ Whether tmux session is running
- ✓/✗ Port connectivity status
- Last few lines of tunnel output

### Stop the tunnel
```bash
bash stop_tunnel.sh
```

Or manually:
```bash
tmux kill-session -t ssh-tunnel
```

## Tmux Commands

### Attach to tunnel session (view output)
```bash
tmux attach -t ssh-tunnel
```

### List all tmux sessions
```bash
tmux list-sessions
```

### Create new window in session
```bash
tmux new-window -t ssh-tunnel
```

### Kill the session
```bash
tmux kill-session -t ssh-tunnel
```

### Detach from session (without killing it)
```bash
# Inside tmux, press:
Ctrl+B, D
```

## How It Works

1. **start_tunnel_simple.sh** - Creates a new tmux session with SSH tunnel
2. **check_tunnel.sh** - Checks if tunnel is running and ports are accessible
3. **stop_tunnel.sh** - Kills the tmux session and stops the tunnel

The SSH tunnel runs in foreground inside the tmux session, with these options:
- `-L 8000:127.0.0.1:8000` - Forward local port 8000 to remote 127.0.0.1:8000
- `-L 8127:127.0.0.1:8127` - Forward local port 8127 to remote 127.0.0.1:8127
- `-N` - Don't execute remote command
- `-o ServerAliveInterval=60` - Send keepalive every 60 seconds
- Auto-reconnects if connection drops

## Usage with Training

Once the tunnel is running, you can directly use the services:
- Retriever: `http://127.0.0.1:8000/retrieve`
- Analyzer: `http://127.0.0.1:8127/generate`

The training scripts are already configured to use these addresses:
```bash
export RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export SELF_OPD_ANALYZER_URL=http://127.0.0.1:8127
```

### Start training (tunnel must be running)
```bash
bash run_tracecredit_nodeskill_opd_smoke.sh
# or
bash scripts/run_tracecredit_nodeskill_opd_formal_entropy.sh
```

## Troubleshooting

### Tunnel not connecting
1. Check SSH password is correct
2. Verify network connectivity: `ping 172.18.167.248`
3. Check if services are running on remote host
4. View tmux session output: `tmux attach -t ssh-tunnel`

### Ports already in use
If you get "Address already in use" error:
```bash
# Kill any existing tunnels
pkill -f "ssh.*8000.*8127"
sleep 1
# Try again
bash start_tunnel_simple.sh
```

### Services not responding
Check both ports are accessible:
```bash
curl http://127.0.0.1:8000/retrieve -d '{}' -H "Content-Type: application/json"
curl http://127.0.0.1:8127/generate -d '{}' -H "Content-Type: application/json"
```

## File Reference

- **start_tunnel_simple.sh** - Start SSH tunnel in tmux (long-running)
- **check_tunnel.sh** - Check tunnel status
- **stop_tunnel.sh** - Stop the tunnel
- **tunnel_daemon_simple.sh** - Alternative daemon script (not used by default)
- **setup_ssh_tunnel.sh** - Original one-time tunnel script (deprecated)
- **stop_ssh_tunnel.sh** - Stop one-time tunnel (deprecated)

## Advanced: Using SSH Keys (No Password)

For even better long-term stability, you can configure SSH public key authentication:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy public key to remote host
ssh-copy-id -i ~/.ssh/id_ed25519.pub luwa@172.18.167.248

# Test key-based login (should not prompt for password)
ssh -i ~/.ssh/id_ed25519 luwa@172.18.167.248 echo "Success!"

# Then update start_tunnel_simple.sh to use:
# ssh -i ~/.ssh/id_ed25519 -L ... (add -i option)
```
