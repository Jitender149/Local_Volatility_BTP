from imports import *

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

print('Simulation is running on:')
print(f'device = {device}')
print()

config = {
    "device": device,
    "data_type": torch.float32,
    "data_type_nn": torch.float32,
    "show": False,
    "mc_load": False,
    "t_": np.linspace(0, 1, 1024),
    "x_": np.linspace(0, 2.5, 1024),
    "S_0": torch.reshape(torch.tensor([1.0], dtype=torch.float32), (-1,1)).to(device),
    "r": torch.tensor(0.02, dtype=torch.float32).to(device),  # short interest rate
    "d": 1,  # dimension of the random variable
    "M": 10**6,  # number of samples
    "N_t": 150,  # number of time steps
    "dt": 0.01,  # size of time step
    "N": 100,  # number of maturities
    "m": 100,  # number of strikes per maturity
    "times": np.linspace(0.5,1.5,100),
    "strikes": np.linspace(0.5, 2.5, 100),
    # Neural Network parameters
    "nn_params": {
        "input_size": 2,
        "output_size": 1,
        "hidden_layers": 4,
        "num_neurons": 64,
        "neurons_per_layer": [64]*4,
        "activation": nn.SiLU,
        "use_batch_norm": False,
        "dropout_rate": False,
        "learning_rate": 1e-3,
        "num_epochs": 45000,
        "print_epochs": 250,
        "save_epochs": 1000,
    },
    # Loss weights
    "lambda_fit": torch.tensor(1.0, dtype=torch.float32),
    "lambda_pde": torch.tensor(1.0, dtype=torch.float32),
    "lambda_arb": torch.tensor(1.0, dtype=torch.float32),
    "lambda_reg": torch.tensor(1.0, dtype=torch.float32),
    "lambda_ini": torch.tensor(1.0, dtype=torch.float32),
    # Boundary condition epsilon
    "boundary_epsilon": 1e-3
}
