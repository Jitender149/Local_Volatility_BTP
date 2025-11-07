# Neural Network for Put Option Pricing on SPX

This repository contains a TensorFlow-based neural network script designed to model and price put options for SPX. The model employs a physics-based loss function approach to ensure consistent and arbitrage-free option prices. Key functionalities include Monte Carlo estimation, custom loss functions based on financial modeling constraints, and configurable parameters for flexibility in pricing.

## Features

- **Custom Physics-Based Loss Functions**: Implements `loss_phi_cal` and `loss_dupire_cal` to enforce financial constraints and ensure consistency with the Dupire equation.
- **Monte Carlo Estimation**: Utilizes Monte Carlo methods for computing option prices across maturity and strike pairs.
- **Lambda-Weighted Training Step**: Combines custom loss functions using lambda weights, allowing flexibility in balancing different loss components.
- **Configurable Pricing Parameters**: Adjust ranges for strike prices and maturities for precise control over option pricing.

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
   git clone https://github.com/your-username/nn-put-option-pricing-spx.git
   cd nn-put-option-pricing-spx
   ```

2. Run the script:
   ```bash
   python tf_NN_put_SPX.py
   ```

The script will train the neural network using custom loss functions, applying automatic differentiation to optimize the model for put option pricing.

### Key Parameters

- **Strike and Maturity Ranges**: Configurable in the script as `k_min`, `k_max`, `t_min`, and `t_max`.
- **Physics-Based Loss Functions**: `loss_phi_cal` and `loss_dupire_cal` ensure consistency with financial constraints.
- **Lambda Weights**: Allows fine-tuning of the balance between different loss components during training.

## Code Structure

- `tf_NN_put_SPX.py`: The main script for setting up, training, and testing the neural network for SPX put option pricing.

### Example

The output will display the training progress and the predicted option prices. The script provides real-time monitoring of losses and pricing accuracy.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.

---

This README file follows the requested structure and style. Adjust the repository URL and other details as needed.
# tf_NN_put_SPX

This repository contains the `tf_NN_put_SPX.py` script, a neural network implementation in TensorFlow designed to predict the price of SPX put options. The script incorporates custom loss functions based on financial modeling constraints, making it a specialized tool for option pricing.

## Features

- **Neural Network for Option Pricing**: Uses a neural network model to predict put option prices based on SPX data.
- **Custom Loss Functions**: Implements physics-based loss functions, `loss_phi_cal` and `loss_dupire_cal`, to improve model accuracy by incorporating constraints based on financial modeling.
- **Monte Carlo Estimation**: Utilizes Monte Carlo simulations to compute option price expectations.
- **Configurable Parameters**: Allows customization of option pricing parameters, including maturities, strike prices, and model configuration.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- TensorFlow 2.x
- Required libraries: `numpy`, `pandas`, `matplotlib` (for any visualization if included)

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/tf_NN_put_SPX.git
cd tf_NN_put_SPX
pip install -r requirements.txt
```

### Usage

Run the script using:

```bash
python tf_NN_put_SPX.py
```

Make sure to update any parameters within the script to align with your specific data or configuration.

### Script Parameters

Adjust parameters for maturity (T), strike (K), and other model parameters within the script as needed:

- **t_min** and **t_max**: Define the range for maturities.
- **k_min** and **k_max**: Set the range for strike prices.
- **Lambda Weights**: Adjust lambda weights in the training step for custom loss functions.

## File Structure

- `tf_NN_put_SPX.py`: Main script for put option pricing.
- `README.md`: Project documentation.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Contributing

Feel free to open issues or submit pull requests to enhance the functionality.

## Contact

For questions, feel free to reach out or open an issue on GitHub.