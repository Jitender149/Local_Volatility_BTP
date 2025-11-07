
# Neural Network for Dupire's Equation

This repository contains the implementation of a neural network framework for solving the Dupire equation. The repository consists of several Python files that handle different tasks, such as data scaling, neural network definitions, configuration, and training scripts.

## Files Overview

- **scale.py**: This file contains utilities for scaling data, such as feature normalization or inverse scaling.
  
- **config.py**: Holds the configuration parameters for the neural network training and testing, including hyperparameters, device settings, and others.

- **mc.py**: Implements Monte Carlo-based utilities for sampling or evaluation related to the neural network's performance.

- **NN.py**: Defines the neural network architecture used for solving the Dupire equation. Includes forward passes and loss function calculations.

- **run.py**: This script acts as the main entry point for running the neural network training or inference process. It orchestrates the interaction between the models, datasets, and training pipeline.

- **NNtrain.py**: Handles the training logic for the neural network. It manages the optimization, loss tracking, and printing of progress throughout the training epochs.

- **imports.py**: Centralized imports for the project. This file includes commonly used libraries such as NumPy, PyTorch, and Matplotlib.

## How to Run the Project

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the parameters in `config.py` based on your environment and data.

4. To start the training process, run:

   ```bash
   python run.py
   ```

   This script will initialize the neural network and begin the training loop using the defined configuration and dataset.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

You can install the required dependencies using the `requirements.txt` file.

## Project Structure

```bash
.
├── scale.py           # Scaling utilities
├── config.py          # Configuration settings
├── mc.py              # Monte Carlo utilities
├── NN.py              # Neural network architecture definition
├── run.py             # Main entry point for training/inference
├── NNtrain.py         # Training process implementation
├── imports.py         # Common imports for the project
└── README.md          # Project documentation
```

## Usage Example

Modify the configuration in `config.py` based on your dataset or desired parameters. You can adjust the neural network architecture, learning rates, optimizer, and other training-related settings.

Once the configuration is set, simply run the `run.py` script to begin training the model:

```bash
python run.py
```

The model's progress, including loss values and intermediate results, will be printed to the console at regular intervals.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
