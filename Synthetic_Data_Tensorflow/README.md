# Local Volatility Model with Neural Networks | Synthetic Data Generated via Monte Carlo

This project implements a deep learning model to estimate local volatility and price options using Monte Carlo simulation and physics-informed neural networks (PINNs). The model is developed in TensorFlow and trained using custom loss functions that incorporate market constraints like the Dupire equation. tf_NN_call_MC.py is trained over a 10 by 20 grid and tf_NN_call_MC_small.py is trained over a 3 by 6 grid.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Parameters](#parameters)
- [Training and Testing](#training-and-testing)
- [Results and Visualization](#results-and-visualization)
- [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/local-volatility-model.git
   cd local-volatility-model
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure TensorFlow can access GPU resources if available.

## Project Structure

- **main()**: Primary function to initialize data, model, and training for each Dupire loss coefficient.
- **LoadData Class**: Handles loading, saving, and processing of datasets, as well as Monte Carlo simulations.
- **PhysicsModel Class**: Constructs neural network models for predicting local volatility and option prices, and defines custom loss functions.
- **Plotter Class**: Creates 3D surface plots and loss plots for visualizing training progress and model performance.
- **Trainer Class**: Runs the training loop, periodically saves model checkpoints, and monitors training loss.

## Usage

Run the main program with:
```python
python tf_NN_call_MC.py
```

This script will train the model using a range of Dupire loss coefficients specified in `ldups`.

## Parameters

- **NN Parameters**:
  - `num_res_blocks`: Number of residual blocks in the neural network.
  - `gaussian_phi`, `gaussian_eta`: Gaussian noise parameters for regularization.
  - `lr`: Learning rate for neural network training.

- **Monte Carlo Parameters**:
  - `N`, `m`: Number of strikes and maturities for option pricing.
  - `S0`: Initial spot price.
  - `r_`: Risk-free rate.
  - `M`: Number of samples for Monte Carlo.
  - `N_t`, `dt`: Number and size of time steps.

- **Training Parameters**:
  - `num_epochs`: Number of epochs to train the model.
  - `print_epochs`: Interval for logging progress.
  - `save_epochs`: Interval for saving model checkpoints.

## Training and Testing

To start training and evaluation:
1. Run the script with each value in `ldups`.
2. Training logs, model checkpoints, and generated plots will be saved in corresponding timestamped directories.

## Results and Visualization

The script generates 3D surface plots for:
- Neural option prices vs. exact option prices
- Local volatility estimates vs. exact local volatility

Loss metrics and training errors are also saved as plots for further analysis.

## License

This project is licensed under the MIT License.