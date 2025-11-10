#!/bin/bash

# Script to run Sigma-Price Network training with nohup
# Results will be stored in the ImpliedVolatility folder

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# Activate virtual environment if it exists
if [ -f "${PARENT_DIR}/poorpeople-env-py310/bin/activate" ]; then
	source "${PARENT_DIR}/poorpeople-env-py310/bin/activate"
	echo "Activated virtual environment: ${PARENT_DIR}/poorpeople-env-py310"
elif [ -f "${PARENT_DIR}/venv/bin/activate" ]; then
	source "${PARENT_DIR}/venv/bin/activate"
	echo "Activated virtual environment: ${PARENT_DIR}/venv"
fi

# Change to the ImpliedVolatility directory
cd "$SCRIPT_DIR"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
LOG_FILE="training_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Log file: $LOG_FILE"
echo "Results will be stored in: $SCRIPT_DIR"
echo "Python: $(which python)"
echo ""

# Run training with nohup
nohup python tf_sigma_price_network.py > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!

echo "Training started with PID: $PID"
echo "To monitor progress, use: tail -f $LOG_FILE"
echo "To check if process is running: ps -p $PID"
echo ""
echo "Training is running in the background. Results will be saved in:"
echo "  - Model checkpoints: ${SCRIPT_DIR}/sigma_price_network_small_*/"
echo "  - Log file: ${SCRIPT_DIR}/${LOG_FILE}"
