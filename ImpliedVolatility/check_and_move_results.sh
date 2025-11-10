#!/bin/bash
# Check training status and move results to main folder when complete

WORKTREE_DIR="/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility"
MAIN_DIR="/home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility"

echo "=== Checking Training Status and Results ==="
echo ""

# Check if training is running
if pgrep -f tf_implied_vol_small.py > /dev/null; then
    PID=$(pgrep -f tf_implied_vol_small.py | head -1)
    echo "✅ Training is RUNNING (PID: $PID)"
    ps aux | grep $PID | grep -v grep | awk '{print "   CPU:", $3"%", "| MEM:", $4"%", "| Runtime:", $10}'
    echo ""
    echo "⏳ Training is still in progress. Results will be available when complete."
    echo ""
    echo "To check results location:"
    echo "  ls -lth $WORKTREE_DIR/implied_vol_small_dataset_*"
else
    echo "✅ Training is COMPLETE (or not running)"
    echo ""
    
    # Find all results directories in worktree
    RESULTS_DIRS=$(ls -td ${WORKTREE_DIR}/implied_vol_small_dataset_* 2>/dev/null)
    
    if [ -n "$RESULTS_DIRS" ]; then
        echo "Found result directories in worktree location:"
        echo "$RESULTS_DIRS" | while read dir; do
            echo "  - $(basename $dir)"
        done
        echo ""
        
        read -p "Move all results to main folder? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Moving results..."
            
            # Move each results directory
            echo "$RESULTS_DIRS" | while read dir; do
                if [ -d "$dir" ]; then
                    BASENAME=$(basename "$dir")
                    if [ ! -d "$MAIN_DIR/$BASENAME" ]; then
                        mv "$dir" "$MAIN_DIR/"
                        echo "  ✅ Moved: $BASENAME"
                    else
                        echo "  ⚠️  Already exists in main folder: $BASENAME"
                    fi
                fi
            done
            
            # Move training log
            if [ -f "${WORKTREE_DIR}/training_output.log" ]; then
                LOG_NAME="training_output_$(date +%Y%m%d_%H%M%S).log"
                mv "${WORKTREE_DIR}/training_output.log" "$MAIN_DIR/$LOG_NAME"
                echo "  ✅ Moved training log: $LOG_NAME"
            fi
            
            echo ""
            echo "✅ All results moved to: $MAIN_DIR/"
            echo ""
            echo "Results are now in main folder:"
            ls -lth "$MAIN_DIR"/implied_vol_small_dataset_* 2>/dev/null | head -5
        else
            echo "Results not moved. They remain in worktree location."
        fi
    else
        echo "No results directories found in worktree location."
        echo ""
        echo "Checking main folder for existing results:"
        ls -lth "$MAIN_DIR"/implied_vol_small_dataset_* 2>/dev/null | head -5 || echo "  No results in main folder yet."
    fi
fi
