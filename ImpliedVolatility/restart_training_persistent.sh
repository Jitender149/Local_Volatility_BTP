#!/bin/bash
# Restart training with nohup (no sudo required)

SCRIPT_DIR="/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility"
cd "$SCRIPT_DIR"

echo "=== Restarting training with nohup (persistent) ==="
echo ""

# Stop current training if running
if pgrep -f tf_implied_vol_small.py > /dev/null; then
    echo "Stopping current training process..."
    pkill -f tf_implied_vol_small.py
    sleep 2
    echo "✅ Stopped."
fi

# Start training with nohup (will persist after terminal closes)
echo "🚀 Starting training with nohup..."
cd "$SCRIPT_DIR"

nohup /home/tanveer/miniconda3/envs/poorpeople-env-py310/bin/python tf_implied_vol_small.py > training_output.log 2>&1 &

TRAINING_PID=$!
sleep 2

# Verify it's running
if ps -p $TRAINING_PID > /dev/null; then
    echo "✅ Training started successfully!"
    echo "   PID: $TRAINING_PID"
    echo ""
    echo "📋 Useful commands:"
    echo "   Check if running:  ps aux | grep tf_implied_vol"
    echo "   View logs:         tail -f $SCRIPT_DIR/training_output.log"
    echo "   Stop training:     kill $TRAINING_PID"
    echo "   Find and stop:     pkill -f tf_implied_vol_small.py"
    echo ""
    echo "💡 You can now close your terminal - training will continue!"
    echo "   The process will keep running even after you close your laptop."
else
    echo "❌ Failed to start training. Check training_output.log for errors."
    exit 1
fi
