#!/bin/bash
# Start Local Volatility PINN 2-Step training from main folder

SCRIPT_DIR="/home/tanveer/poorpeople/LocalVolatility/LocalVolatilityPINN"
cd "$SCRIPT_DIR"

echo "=== Starting Local Volatility PINN 2-Step Training ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if training is already running
if pgrep -f tf_local_vol_pinn_2step.py > /dev/null; then
    echo "⚠️  Training is already running!"
    ps aux | grep python | grep tf_local_vol_pinn_2step | grep -v grep
    exit 1
fi

# Start training with nohup
echo "🚀 Starting training..."
nohup /home/tanveer/miniconda3/envs/poorpeople-env-py310/bin/python tf_local_vol_pinn_2step.py > training_output_2step.log 2>&1 &

TRAINING_PID=$!
sleep 2

# Verify it's running
if ps -p $TRAINING_PID > /dev/null; then
    echo "✅ Training started successfully!"
    echo "   PID: $TRAINING_PID"
    echo "   Working directory: $SCRIPT_DIR"
    echo ""
    echo "📋 Useful commands:"
    echo "   Check status:  ps aux | grep tf_local_vol_pinn_2step"
    echo "   View logs:     tail -f $SCRIPT_DIR/training_output_2step.log"
    echo "   Stop training: kill $TRAINING_PID"
    echo ""
    echo "💡 Results will be saved in:"
    echo "   $SCRIPT_DIR/local_vol_pinn_2step_YYYY_MM_DD_HH_MM/"
    echo ""
    echo "💡 You can now close your session - training will continue!"
else
    echo "❌ Failed to start training. Check training_output_2step.log for errors."
    exit 1
fi

