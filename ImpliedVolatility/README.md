# Implied Volatility Training

This directory contains the training script for learning implied volatility surfaces using a two-network architecture (SigmaNet + PriceNet) on a small dataset (3×6 option prices).

## Architecture

The model consists of two neural networks:

1. **SigmaNet**: Estimates implied volatility σ(K, T) from strike K and maturity T
   - Input: (K, T) - 2D
   - Architecture: Dense layers with ReLU activation
   - Output: σ (softplus activation to ensure σ > 0)

2. **PriceNet**: Estimates option price P(K, T, σ) using strike, maturity, and implied volatility
   - Input: (K, T, σ) - 3D
   - Architecture: Dense layers with ReLU activation
   - Output: Option price (linear activation)

## Dataset

- **Size**: 3×6 = 18 option prices
  - 3 maturities
  - 6 strikes per maturity
- **Parameters**:
  - Spot price S₀ = 1000
  - Risk-free rate r = 0.04
  - Strike range: [500, 3000]

## Training

Run the training script:
```bash
python tf_implied_vol_small.py
```

### Training Parameters

- **Epochs**: 30,000
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: 
  - Data loss: MSE on option prices
  - Smoothness regularization: L2 on second derivatives of σ w.r.t. K and T
  - Regularization weight: λ_smooth = 1e-3

### Output Files

The script generates:
- `sigma_net_{epoch}.keras` - Saved SigmaNet models
- `price_net_{epoch}.keras` - Saved PriceNet models
- `volatility_surface_{epoch}.png` - 3D plots of learned volatility and price surfaces
- `training_curves_{epoch}.png` - Loss and RMSE curves
- `prediction_comparison_{epoch}.png` - Comparison of predictions with training data
- `training_history.csv` - Training metrics

## Model Architecture Details

### SigmaNet
- Hidden layers: [64, 64, 64]
- Activation: ReLU
- Output activation: Softplus (ensures σ > 0)
- Regularization: L2 (1e-5)

### PriceNet
- Hidden layers: [128, 128, 128]
- Activation: ReLU
- Output activation: Linear
- Regularization: L2 (1e-5)

## Comparison with Original Approach

This implementation differs from the original `tf_NN_call_MC_small.py`:

1. **Architecture**: Two separate networks (SigmaNet + PriceNet) instead of residual blocks
2. **Loss Function**: Includes smoothness regularization on volatility surface
3. **Training**: Simpler training loop without physics-informed constraints

