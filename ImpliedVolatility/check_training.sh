#!/bin/bash
# Quick script to check training status
echo "=== Training Status Check ==="
echo ""
ps aux | grep python | grep tf_implied_vol_small | grep -v grep | awk '{print "PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| Runtime:", $10}' || echo "❌ Training is not running"
echo ""
echo "=== Recent Log Output ==="
tail -20 training_output.log 2>/dev/null | tail -10
echo ""
echo "=== Result Directories ==="
ls -lth implied_vol_small_dataset_* 2>/dev/null | head -3 || echo "No results yet"
