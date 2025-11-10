# Local Volatility PINN - Bae, Kang & Lee (2021)

This directory contains the implementation of a Physics-Informed Neural Network (PINN) for learning local volatility surfaces, based on the approach by Bae, Kang & Lee (2021).

## Architecture

The model consists of two neural networks:

1. **SigmaNet**: Estimates local volatility `σ(S, t)` from normalized spot price `S` and time `t`
   - Input: `(S, t)` - 2D (normalized S/S0, time)
   - Architecture: Dense layers [64, 64, 64] with Tanh activation
   - Output: `σ` (softplus activation to ensure σ > 0)

2. **PriceNet**: Estimates option price `V(S, t)` from normalized spot price and time
   - Input: `(S, t)` - 2D
   - Architecture: Dense layers [64, 64, 64] with Tanh activation
   - Output: Option price `V` (linear activation)

## Physics-Informed Loss

The model enforces the Black-Scholes PDE:

```
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

### Loss Components:

1. **Data Loss**: MSE on synthetic option prices
2. **PDE Loss**: Black-Scholes PDE residual at collocation points
3. **Boundary Condition Loss**: 
   - Lower boundary (S=0): V = 0
   - Upper boundary (S=S_max): V ≈ S - e^{-r(T-t)}K
4. **Initial Condition Loss**: Terminal payoff V(S, T) = max(S - K, 0)

## Training

### Start Training:
```bash
cd /home/tanveer/poorpeople/LocalVolatility/LocalVolatilityPINN
./start_training.sh
```

### Check Status:
```bash
./check_training.sh
```

### Manual Training:
```bash
python tf_local_vol_pinn.py
```

### Command-Line Arguments:
```bash
python tf_local_vol_pinn.py \
    --lambda_data 1.0 \
    --lambda_pde 1e-4 \
    --lambda_bc 1e-3 \
    --lambda_ic 1e-3 \
    --lr 1e-3 \
    --num_epochs 20000
```

## Parameters

### Default Parameters:
- **Learning Rate**: 1e-3
- **Epochs**: 20,000
- **Loss Weights**:
  - `lambda_data = 1.0` (Data fitting)
  - `lambda_pde = 1e-4` (PDE constraint)
  - `lambda_bc = 1e-3` (Boundary conditions)
  - `lambda_ic = 1e-3` (Initial condition)

### Data Generation:
- **Data Points**: 1,000 (synthetic option prices)
- **Collocation Points**: 2,000 (for PDE residual)
- **Boundary Points**: 200 per boundary (2 boundaries)
- **Initial Condition Points**: 200

### Domain:
- **S range**: [0.0, 3.0] (normalized S/S0)
- **t range**: [0.0, 1.0] (normalized time)

## Output Files

Results are saved in timestamped directories:
```
local_vol_pinn_YYYY_MM_DD_HH_MM/
```

### Model Files:
- `NN_sigma_{epoch}.keras` - SigmaNet models (saved at epochs 5k, 10k, 15k, 20k, final)
- `NN_price_{epoch}.keras` - PriceNet models

### Plot Files:
- `surfaces_{epoch}.png` - 3D plots of learned and exact price/volatility surfaces
- `losses_{epoch}.png` - Training loss curves

## Synthetic Data

The code uses synthetic data generated from a known local volatility function:

```python
σ(S, t) = 0.3 + 0.2 * sin(πS) * exp(-t)
```

This allows for validation against the exact solution.

## Comparison with Other Approaches

### vs. Implied Volatility Model:
- **This model**: Learns local volatility directly from PDE
- **Implied Vol model**: Learns implied volatility from option prices

### vs. Original Local Vol Model:
- **This model**: Uses PINN approach with synthetic data
- **Original model**: Uses Monte Carlo simulation with Dupire equation

## References

- Bae, Kang & Lee (2021): Physics-Informed Neural Networks for Local Volatility

