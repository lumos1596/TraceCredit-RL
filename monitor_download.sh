#!/usr/bin/env bash
# Monitor checkpoint download progress

CHECKPOINT_DIR="/home/luwa/Documents/Tree-GRPO/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"

echo "Checkpoint Download Monitor"
echo "============================"
echo ""

# Check if download process is running
if pgrep -f "hf download.*treegrpo-rl-opd-handoff" > /dev/null; then
    echo "✓ Download process is running"
    echo ""
    
    # Show memory usage
    PID=$(pgrep -f "hf download.*treegrpo-rl-opd-handoff" | head -1)
    MEM=$(ps -p "$PID" --no-headers -o rss | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "Memory usage: $MEM"
    echo ""
fi

# Show downloaded file size
if [[ -d "$CHECKPOINT_DIR" ]]; then
    SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | awk '{print $1}')
    echo "Downloaded: $SIZE (target: ~36 GB)"
    
    # Estimate time remaining (very rough)
    BYTES=$(du -sb "$CHECKPOINT_DIR" 2>/dev/null | awk '{print $1}')
    TARGET_BYTES=$((36 * 1024 * 1024 * 1024))
    PERCENT=$((100 * BYTES / TARGET_BYTES))
    echo "Progress: $PERCENT%"
else
    echo "Checkpoint directory not found yet"
fi

echo ""
echo "Latest files:"
find "$CHECKPOINT_DIR" -type f 2>/dev/null | sort | tail -5 | xargs -I {} bash -c 'ls -lh "{}" | awk "{print \$5, \$9}"'

echo ""
echo "To continue monitoring, run:"
echo "  watch -n 5 'bash /home/luwa/Documents/Tree-GRPO/monitor_download.sh'"
