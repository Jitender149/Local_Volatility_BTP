#!/bin/bash
# Script to monitor training and move results to main folder when complete

WORKTREE_DIR="/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility"
MAIN_DIR="/home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility"

echo "=== Monitoring Training and Moving Results ==="
echo ""

# Check if training is running
if ! pgrep -f tf_implied_vol_small.py > /dev/null; then
    echo "⚠️  Training is not running. Checking for existing results..."
    
    # Find latest results directory in worktree
    LATEST_RESULTS=$(ls -td ${WORKTREE_DIR}/implied_vol_small_dataset_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_RESULTS" ]; then
        echo "Found results directory: $(basename $LATEST_RESULTS)"
        echo "Moving to main folder..."
        
        # Move results directory
        mv "$LATEST_RESULTS" "$MAIN_DIR/"
        
        # Move training log if exists
        if [ -f "${WORKTREE_DIR}/training_output.log" ]; then
            mv "${WORKTREE_DIR}/training_output.log" "$MAIN_DIR/training_output_$(date +%Y%m%d_%H%M%S).log"
        fi
        
        echo "✅ Results moved to: $MAIN_DIR/$(basename $LATEST_RESULTS)"
    else
        echo "No results directory found."
    fi
else
    echo "Training is still running. Will check again later."
    echo ""
    echo "To manually move results when training completes, run:"
    echo "  cd $WORKTREE_DIR"
    echo "  mv implied_vol_small_dataset_* $MAIN_DIR/"
fi
