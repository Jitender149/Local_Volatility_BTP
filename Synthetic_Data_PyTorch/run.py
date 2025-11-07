from mc import *
from scale import *
from NN import *
from NNtrain import *
from config import *

config["lambda_ini"] = torch.tensor(1e-3, dtype=torch.float32)
config["lambda_pde"] = torch.tensor(1e-1, dtype=torch.float32)
config["boundary_epsilon"] = 0.1

config["mc_load"] = False #Use True to re-use past MC Runs
# config["nn_params"]["num_epochs"] = 1

for key, value in config.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            locals()[sub_key] = sub_value
    else:
        locals()[key] = value

nn_params = [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate]
scheduler = None
# scheduler = [StepLR, {'step_size': lr_drop, 'gamma': lr_drop_value}]
# scheduler = [ReduceLROnPlateau, {'mode': 'min', 'factor': lr_drop_value, 'patience': 100, 'verbose': True}]
scheduler = [CosineAnnealingLR, {'T_max': 50}]

identifier = None
# identifier = f'boundary_epsilon_{boundary_epsilon}'
dirname = f'Dupire_fit_{lambda_fit:.1f}_pde_{lambda_pde:.2f}_arb_{lambda_arb:.2f}_reg_{lambda_reg:.1f}_ini_{lambda_ini:.2e}_{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
os.makedirs(dirname, exist_ok=True)

mc = MonteCarloLocalVolatility(S_0, r, x_, t_, d, M, N_t, dt, data_type, show, device, dirname)
if mc_load:
    S_matrix, t_all = torch.tensor(np.load('S_matrix.npy')).to(device), torch.tensor(np.load('t_all.npy')).to(device)
else:
    S_matrix, t_all = mc.run()
    np.save(os.path.join(dirname,f'S_matrix.npy'), S_matrix.cpu().numpy())
    np.save(os.path.join(dirname,f't_all.npy'), t_all.cpu().numpy())

scale = ScaleQuantities(S_0, r, t_all, S_matrix, N, m, strikes, times, data_type, show, device, dirname)
scale.run()

NNTrainer = NNTrain(mc, scale, nn_params, learning_rate, boundary_epsilon, lambda_fit, lambda_pde, lambda_arb, lambda_reg, lambda_ini, num_epochs, print_epochs, save_epochs, dirname, scheduler)
NNTrainer.test()
NNTrainer.run()
