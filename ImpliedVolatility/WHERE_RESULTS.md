# Where to Find Training Results

## Results Location

Training results are saved in a **timestamped subdirectory** within the ImpliedVolatility folder.

### Directory Name Format
```
implied_vol_small_dataset_YYYY_MM_DD_HH_MM/
```

Example:
```
implied_vol_small_dataset_2025_11_07_21_14/
```

### Where to Look

**If training is running from worktree location:**
```
/home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility/implied_vol_small_dataset_YYYY_MM_DD_HH_MM/
```

**If training is running from main folder:**
```
/home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility/implied_vol_small_dataset_YYYY_MM_DD_HH_MM/
```

## What Files Will Be Saved

### Model Files (saved at epochs 10k, 20k, 30k, and final)
- `NN_sigma_10000.keras` - SigmaNet model at epoch 10,000
- `NN_price_10000.keras` - PriceNet model at epoch 10,000
- `NN_sigma_20000.keras` - SigmaNet model at epoch 20,000
- `NN_price_20000.keras` - PriceNet model at epoch 20,000
- `NN_sigma_30000.keras` - SigmaNet model at epoch 30,000
- `NN_price_30000.keras` - PriceNet model at epoch 30,000
- `NN_sigma_final.keras` - Final SigmaNet model
- `NN_price_final.keras` - Final PriceNet model

### Plot Files (saved at epochs 10k, 20k, 30k, and final)
- `phi_10000.png` - Option price surfaces (neural, exact, error)
- `eta_error_weight_10000.png` - Implied volatility surfaces (neural, exact, error)
- `losses_10000.png` - Training loss curves
- `phi_20000.png`, `eta_error_weight_20000.png`, `losses_20000.png`
- `phi_30000.png`, `eta_error_weight_30000.png`, `losses_30000.png`
- `phi_final.png`, `eta_error_weight_final.png`, `losses_final.png`
- `errors_final.png` - Final RMSE and error curves

## How to Check Results

### Find the latest results directory:
```bash
cd /home/tanveer/.cursor/worktrees/LocalVolatility__SSH__172.31.88.252_/uy0kn/ImpliedVolatility
ls -lth implied_vol_small_dataset_* | head -1
```

### Or check from main folder:
```bash
cd /home/tanveer/poorpeople/LocalVolatility/ImpliedVolatility
ls -lth implied_vol_small_dataset_* | head -1
```

### View all saved files:
```bash
cd implied_vol_small_dataset_YYYY_MM_DD_HH_MM/
ls -lh
```

## Training Log

The training log is saved in:
```
ImpliedVolatility/training_output.log
```

This contains all print statements and progress updates.
