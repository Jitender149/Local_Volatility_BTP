#!/bin/bash
# Check status of Local Volatility PINN training

SCRIPT_DIR="/home/tanveer/poorpeople/LocalVolatility/LocalVolatilityPINN"
cd "$SCRIPT_DIR"

echo "=== Training Status ==="
echo ""

# Check if process is running
if pgrep -f tf_local_vol_pinn.py > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    ps aux | grep python | grep tf_local_vol_pinn | grep -v grep
    echo ""
    echo "📋 Recent log output:"
    echo "----------------------------------------"
    tail -30 training_output.log 2>/dev/null || echo "No log file found"
else
    echo "❌ Training is NOT running"
    echo ""
    echo "📋 Last log output:"
    echo "----------------------------------------"
    tail -50 training_output.log 2>/dev/null || echo "No log file found"
fi

echo ""
echo "📁 Results directories:"
ls -lth local_vol_pinn_* 2>/dev/null | head -5 || echo "No results directories found"

