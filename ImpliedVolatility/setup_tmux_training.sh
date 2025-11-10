#!/bin/bash
# Complete script to set up training in tmux

SESSION_NAME="implied_vol_training"
SCRIPT_DIR="/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility"

echo "=== Setting up training in tmux ==="
echo ""

# Step 1: Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed."
    echo "Please install it with: sudo apt-get install tmux"
    echo "Then run this script again."
    exit 1
fi

# Step 2: Stop current training if running
echo "Checking for running training processes..."
if pgrep -f tf_implied_vol_small.py > /dev/null; then
    echo "⚠️  Stopping current training process..."
    pkill -f tf_implied_vol_small.py
    sleep 2
    echo "✅ Stopped."
else
    echo "✅ No running training process found."
fi

# Step 3: Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Session $SESSION_NAME already exists."
    read -p "Kill existing session and start new? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
        echo "✅ Killed existing session."
    else
        echo "Attaching to existing session..."
        tmux attach-session -t $SESSION_NAME
        exit 0
    fi
fi

# Step 4: Start training in tmux
echo "🚀 Starting training in tmux session: $SESSION_NAME"
cd $SCRIPT_DIR

tmux new-session -d -s $SESSION_NAME -c $SCRIPT_DIR \
    "source /home/tanveer/miniconda3/etc/profile.d/conda.sh && \
     conda activate poorpeople-env-py310 && \
     python tf_implied_vol_small.py"

sleep 2

# Check if session is running
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ Training started successfully in tmux!"
    echo ""
    echo "📋 Useful commands:"
    echo "  Attach to session:    tmux attach-session -t $SESSION_NAME"
    echo "  Detach (keep running): Press Ctrl+B, then D"
    echo "  View logs:            tail -f $SCRIPT_DIR/training_output.log"
    echo "  List sessions:        tmux list-sessions"
    echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
    echo ""
    echo "💡 You can now close your laptop - training will continue!"
else
    echo "❌ Failed to start tmux session. Check errors above."
    exit 1
fi
