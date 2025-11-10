#!/bin/bash
# Script to start training in tmux session
# First install tmux: sudo apt-get install tmux

SESSION_NAME="implied_vol_training"
SCRIPT_DIR="/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session: $SESSION_NAME"
    cd $SCRIPT_DIR
    tmux new-session -d -s $SESSION_NAME -c $SCRIPT_DIR \
        "source /home/tanveer/miniconda3/etc/profile.d/conda.sh && \
         conda activate poorpeople-env-py310 && \
         python tf_implied_vol_small.py"
    
    echo "Training started in tmux session: $SESSION_NAME"
    echo ""
    echo "To attach to the session:"
    echo "  tmux attach-session -t $SESSION_NAME"
    echo ""
    echo "To detach from tmux (keep training running):"
    echo "  Press Ctrl+B, then D"
    echo ""
    echo "To view logs:"
    echo "  tail -f $SCRIPT_DIR/training_output.log"
fi
