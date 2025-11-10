# Workflow: Working from Main Folder

## ✅ Setup Complete

From now on, **work from the main LocalVolatility folder**:
```
/home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility/
```

## Current Training Status

⚠️ **Current training is running from worktree location** (started before this setup).

When it completes, use the script to move results:
```bash
cd /home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility
./check_and_move_results.sh
```

## Future Training

**Start training from main folder:**
```bash
cd /home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility
./start_training_main.sh
```

This will:
- Start training from the main folder
- Save results in: `ImpliedVolatility/implied_vol_small_dataset_YYYY_MM_DD_HH_MM/`
- Run with nohup (persists after closing session)

## Available Scripts

1. **`start_training_main.sh`** - Start training from main folder
2. **`check_and_move_results.sh`** - Check training status and move results when complete
3. **`check_training.sh`** - Quick status check
4. **`restart_training_persistent.sh`** - Restart training with nohup

## Results Location

**Results will be saved in:**
```
/home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility/implied_vol_small_dataset_YYYY_MM_DD_HH_MM/
```

**Files saved:**
- Model files: `NN_sigma_{epoch}.keras`, `NN_price_{epoch}.keras`
- Plot files: `phi_{epoch}.png`, `eta_error_weight_{epoch}.png`, `losses_{epoch}.png`
- Final plots: `errors_final.png`

## Moving Existing Results

To move results from worktree to main folder:
```bash
cd /home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility
./check_and_move_results.sh
```

Or manually:
```bash
# From worktree location
cd /home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility
mv implied_vol_small_dataset_* /home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility/
```
