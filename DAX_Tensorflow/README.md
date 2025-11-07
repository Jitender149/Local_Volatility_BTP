# Neural Network for Call Option Pricing on DAX

This repository contains a TensorFlow-based neural network script designed to model and price call options for DAX. The model uses a physics-based loss function approach to provide consistent and arbitrage-free option prices. Key functionalities include dynamic scheduling, automatic differentiation, and financial modeling constraints for accurate and reliable option pricing.

## Features

- **Custom Physics-Based Loss Functions**: Includes `loss_phi_cal` and `loss_dupire_cal` that impose financial constraints and ensure consistency with the Dupire equation.
- **Monte Carlo Estimation**: Utilizes Monte Carlo methods for computing option prices based on maturity and strike pairs.
- **Lambda-Weighted Training Step**: Combines custom loss functions using lambda weights, allowing flexibility in balancing between losses.
- **Cosine Annealing Scheduler**: Adjusts the learning rate dynamically during training for optimized convergence.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy

Install the required packages using:
```bash
pip install tensorflow numpy
```

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nn-call-option-pricing-dax.git
   cd nn-call-option-pricing-dax
   ```

2. Run the script:
   ```bash
   python tf_NN_call_DAX.py
   ```

The script will train the neural network using the provided custom loss functions, applying automatic differentiation and dynamic scheduling.

### Key Parameters

- **Strike and Maturity Ranges**: Configurable in the script as `k_min`, `k_max`, `t_min`, and `t_max`.
- **Physics-Based Loss Functions**: `loss_phi_cal` and `loss_dupire_cal` ensure adherence to financial modeling constraints.
- **Learning Rate Scheduler**: The cosine annealer is applied during training to adjust the learning rate dynamically.

## Code Structure

- `tf_NN_call_DAX.py`: The main script for setting up, training, and testing the neural network for DAX call option pricing.

### Example

The output will display the training progress and eventually the predicted option prices. The script will provide real-time monitoring of losses and pricing accuracy.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.