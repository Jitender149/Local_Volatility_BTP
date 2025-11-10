# ===============================================================
# Local Volatility PINN based on Bae, Kang & Lee (2021)
# PyTorch Implementation
# ===============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import datetime
import argparse

torch.manual_seed(42)
if torch.cuda.is_available():
	torch.cuda.manual_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Simulation is running on: {device}')
print()

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

show = False
num_epochs = 20000
print_epochs = 1000
save_epochs = 5000

# Financial parameters
r = 0.04
S_min, S_max = 0.0, 3.0     # normalized range (S/S0)
t_min, t_max = 0.0, 1.0     # time (0 → maturity)

# Learning rate
lr = 1e-3

# Loss weights (can be overridden via command-line arguments)
lambda_data = 1.0
lambda_pde = 1e-4
lambda_bc = 1e-3
lambda_ic = 1e-3

# Data generation parameters
N_data = 1000
N_colloc = 2000
N_bc = 200
N_ic = 200

# Plotting parameters
N_plot = 100

def parse_arguments():
	"""Parse command-line arguments for weight tuning"""
	parser = argparse.ArgumentParser(description='Local Volatility PINN Training (PyTorch)')
	parser.add_argument('--lambda_data', type=float, default=1.0, help='Data loss weight')
	parser.add_argument('--lambda_pde', type=float, default=1e-4, help='PDE residual weight')
	parser.add_argument('--lambda_bc', type=float, default=1e-3, help='Boundary condition weight')
	parser.add_argument('--lambda_ic', type=float, default=1e-3, help='Initial condition weight')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--num_epochs', type=int, default=20000, help='Number of training epochs')
	args = parser.parse_args()
	return args

# ---------------------------------------------------------------
# SYNTHETIC TRAINING DATA
# ---------------------------------------------------------------

def exact_sigma_local(S, t):
	"""Known local volatility function (synthetic truth for testing)"""
	return 0.3 + 0.2 * torch.sin(np.pi * S) * torch.exp(-t)

def exact_price(S, t):
	"""Dummy target price surface (e.g., precomputed option prices)"""
	# Just for training demonstration; in real setting use Monte Carlo or market data
	sigma = exact_sigma_local(S, t)
	return torch.exp(-0.5 * sigma * S) * torch.clamp(S - 1.0, min=0.0)

# ---------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------

def _plot_surfaces(pinn, dirname, step, S_min, S_max, t_min, t_max, N_plot):
	"""Plot learned price and volatility surfaces"""
	S_grid = np.linspace(S_min, S_max, N_plot)
	t_grid = np.linspace(t_min, t_max, N_plot)
	S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)
	inp = np.stack([S_mesh.flatten(), t_mesh.flatten()], axis=1)
	inp_tensor = torch.tensor(inp, dtype=torch.float32, device=device)
	
	with torch.no_grad():
		V_pred = pinn.NN_price(inp_tensor).cpu().numpy().reshape(N_plot, N_plot)
		sigma_pred = pinn.NN_sigma(inp_tensor).cpu().numpy().reshape(N_plot, N_plot)
	
	# Exact surfaces for comparison
	V_exact = exact_price(torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, device=device),
						  torch.tensor(t_mesh.flatten()[:, None], dtype=torch.float32, device=device)).cpu().numpy().reshape(N_plot, N_plot)
	sigma_exact = exact_sigma_local(torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, device=device),
									torch.tensor(t_mesh.flatten()[:, None], dtype=torch.float32, device=device)).cpu().numpy().reshape(N_plot, N_plot)
	
	# Price surfaces
	fig = plt.figure(figsize=(24, 8), dpi=450)
	
	ax = fig.add_subplot(2, 3, 1, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, V_pred, cmap=cm.coolwarm, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("V(S,t)")
	ax.set_title("Learned Option Price Surface")
	
	ax = fig.add_subplot(2, 3, 2, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, V_exact, cmap=cm.coolwarm, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("V(S,t)")
	ax.set_title("Exact Option Price Surface")
	
	ax = fig.add_subplot(2, 3, 3, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, V_pred - V_exact, cmap=cm.RdBu_r, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("Error")
	ax.set_title("Price Error Surface")
	
	# Volatility surfaces
	ax = fig.add_subplot(2, 3, 4, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, sigma_pred, cmap=cm.viridis, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("σ_loc(S,t)")
	ax.set_title("Learned Local Volatility Surface")
	
	ax = fig.add_subplot(2, 3, 5, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, sigma_exact, cmap=cm.viridis, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("σ_loc(S,t)")
	ax.set_title("Exact Local Volatility Surface")
	
	ax = fig.add_subplot(2, 3, 6, projection="3d")
	ax.plot_surface(S_mesh, t_mesh, sigma_pred - sigma_exact, cmap=cm.RdBu_r, linewidth=0)
	ax.set_xlabel("S")
	ax.set_ylabel("t")
	ax.set_zlabel("Error")
	ax.set_title("Volatility Error Surface")
	
	plt.tight_layout()
	plt.savefig(os.path.join(dirname, f'surfaces_{step}.png'))
	if show:
		plt.show()
	else:
		plt.close()

def _plot_losses(loss_history, loss_data_history, loss_pde_history, 
				loss_bc_history, loss_ic_history, dirname, step):
	"""Plot training loss curves"""
	fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=450)
	
	# Total loss
	ax[0].semilogy(loss_history, label='Total Loss', linewidth=1)
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel('Loss')
	ax[0].set_title('Total Loss')
	ax[0].legend()
	ax[0].grid(True, alpha=0.3)
	
	# Component losses
	ax[1].semilogy(loss_data_history, label='Data Loss', linewidth=1)
	ax[1].semilogy(loss_pde_history, label='PDE Loss', linewidth=1)
	ax[1].semilogy(loss_bc_history, label='BC Loss', linewidth=1)
	ax[1].semilogy(loss_ic_history, label='IC Loss', linewidth=1)
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Loss')
	ax[1].set_title('Component Losses')
	ax[1].legend()
	ax[1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(dirname, f'losses_{step}.png'))
	if show:
		plt.show()
	else:
		plt.close()

def main():
	# Parse command-line arguments
	args = parse_arguments()
	global lambda_data, lambda_pde, lambda_bc, lambda_ic, lr, num_epochs
	lambda_data = args.lambda_data
	lambda_pde = args.lambda_pde
	lambda_bc = args.lambda_bc
	lambda_ic = args.lambda_ic
	lr = args.lr
	num_epochs = args.num_epochs

	print('Starting Local Volatility PINN Training (PyTorch)')
	print('')
	print(f'Loss weights: lambda_data={lambda_data}, lambda_pde={lambda_pde}, lambda_bc={lambda_bc}, lambda_ic={lambda_ic}')
	print(f'Learning rate: {lr}, Epochs: {num_epochs}')
	print()

	# Create results directory
	identifier = f'local_vol_pinn_pytorch'
	folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
	dirname = folder_name_save
	os.makedirs(dirname, exist_ok=True)

	# Create collocation & data points
	print('Generating training data...')
	S_data = torch.rand(N_data, 1, device=device) * (S_max - S_min) + S_min
	t_data = torch.rand(N_data, 1, device=device) * (t_max - t_min) + t_min
	V_data = exact_price(S_data, t_data)

	# Boundary / initial conditions
	# Initial condition: at maturity t=T (normalized t=1) → payoff
	S_ic = torch.linspace(S_min, S_max, N_ic, device=device).unsqueeze(1)
	t_ic = torch.ones_like(S_ic) * t_max
	V_ic = torch.clamp(S_ic - 1.0, min=0.0)  # (S - K)+ payoff, K=1 normalized

	# Lower boundary: S=0 → V=0
	S_bc0 = torch.zeros(N_bc, 1, device=device)
	t_bc0 = torch.linspace(t_min, t_max, N_bc, device=device).unsqueeze(1)
	V_bc0 = torch.zeros_like(S_bc0)

	# Upper boundary: linear asymptote V ≈ S - e^{-r(T-t)}K
	S_bc1 = torch.ones(N_bc, 1, device=device) * S_max
	t_bc1 = torch.linspace(t_min, t_max, N_bc, device=device).unsqueeze(1)
	V_bc1 = S_bc1 - torch.exp(-r * (1 - t_bc1))

	# Collocation points (for PDE residual)
	S_colloc = torch.rand(N_colloc, 1, device=device) * (S_max - S_min) + S_min
	t_colloc = torch.rand(N_colloc, 1, device=device) * (t_max - t_min) + t_min

	print(f'Data points: {N_data}')
	print(f'Collocation points: {N_colloc}')
	print(f'Boundary condition points: {N_bc * 2}')
	print(f'Initial condition points: {N_ic}')
	print()

	# ---------------------------------------------------------------
	# NEURAL NETWORKS: SigmaNet and PriceNet
	# ---------------------------------------------------------------

	class SigmaNet(nn.Module):
		def __init__(self):
			super(SigmaNet, self).__init__()
			self.net = nn.Sequential(
				nn.Linear(2, 64),
				nn.Tanh(),
				nn.Linear(64, 64),
				nn.Tanh(),
				nn.Linear(64, 64),
				nn.Tanh(),
				nn.Linear(64, 1),
				nn.Softplus()
			)
		
		def forward(self, x):
			return self.net(x)

	class PriceNet(nn.Module):
		def __init__(self):
			super(PriceNet, self).__init__()
			self.net = nn.Sequential(
				nn.Linear(2, 64),
				nn.Tanh(),
				nn.Linear(64, 64),
				nn.Tanh(),
				nn.Linear(64, 64),
				nn.Tanh(),
				nn.Linear(64, 1)
			)
		
		def forward(self, x):
			return self.net(x)

	# ---------------------------------------------------------------
	# PHYSICS-INFORMED MODEL
	# ---------------------------------------------------------------

	class LocalVolPINN(nn.Module):
		def __init__(self, r):
			super(LocalVolPINN, self).__init__()
			self.r = r
			self.NN_sigma = SigmaNet().to(device)
			self.NN_price = PriceNet().to(device)

		# ------------------------
		# PDE residual (Dupire/Black-Scholes)
		# ------------------------
		def loss_pde(self, S, t):
			# Enable gradient computation
			S.requires_grad_(True)
			t.requires_grad_(True)
			
			# Concatenate inputs
			inp = torch.cat([S, t], dim=1)
			V = self.NN_price(inp)
			sigma = self.NN_sigma(inp)
			
			# Compute first derivatives
			V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
			V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
			
			# Compute second derivative
			V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True, retain_graph=True)[0]
			
			# Black-Scholes PDE: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
			pde_res = V_t + 0.5 * sigma**2 * S**2 * V_SS + self.r * S * V_S - self.r * V
			
			return torch.mean(pde_res**2)

		# ------------------------
		# Total loss
		# ------------------------
		def total_loss(self, data, bc, ic):
			(S_d, t_d, V_d) = data
			(S_bc0, t_bc0, V_bc0, S_bc1, t_bc1, V_bc1) = bc
			(S_ic, t_ic, V_ic) = ic

			# Data loss
			inp_data = torch.cat([S_d, t_d], dim=1)
			V_pred_data = self.NN_price(inp_data)
			L_data = torch.mean((V_pred_data - V_d)**2)

			# PDE residual
			L_pde = self.loss_pde(S_colloc, t_colloc)

			# Boundary losses
			inp_bc0 = torch.cat([S_bc0, t_bc0], dim=1)
			inp_bc1 = torch.cat([S_bc1, t_bc1], dim=1)
			V_pred_bc0 = self.NN_price(inp_bc0)
			V_pred_bc1 = self.NN_price(inp_bc1)
			L_bc0 = torch.mean((V_pred_bc0 - V_bc0)**2)
			L_bc1 = torch.mean((V_pred_bc1 - V_bc1)**2)

			# Initial (terminal payoff) loss
			inp_ic = torch.cat([S_ic, t_ic], dim=1)
			V_pred_ic = self.NN_price(inp_ic)
			L_ic = torch.mean((V_pred_ic - V_ic)**2)

			total = (
				lambda_data * L_data +
				lambda_pde * L_pde +
				lambda_bc * (L_bc0 + L_bc1) +
				lambda_ic * L_ic
			)

			return total, (L_data, L_pde, L_bc0 + L_bc1, L_ic)

	# ---------------------------------------------------------------
	# TRAINING LOOP
	# ---------------------------------------------------------------

	pinn = LocalVolPINN(r)
	optimizer = torch.optim.Adam(list(pinn.NN_sigma.parameters()) + list(pinn.NN_price.parameters()), lr=lr)
	
	data = (S_data, t_data, V_data)
	bc = (S_bc0, t_bc0, V_bc0, S_bc1, t_bc1, V_bc1)
	ic = (S_ic, t_ic, V_ic)

	loss_history = []
	loss_data_history = []
	loss_pde_history = []
	loss_bc_history = []
	loss_ic_history = []

	print("Training Local Volatility PINN ...")
	print()

	for epoch in range(num_epochs + 1):
		optimizer.zero_grad()
		total_loss, components = pinn.total_loss(data, bc, ic)
		total_loss.backward()
		optimizer.step()

		loss_history.append(total_loss.item())
		loss_data_history.append(components[0].item())
		loss_pde_history.append(components[1].item())
		loss_bc_history.append(components[2].item())
		loss_ic_history.append(components[3].item())

		if epoch % print_epochs == 0:
			print(f"Epoch {epoch:05d} | Total Loss = {total_loss.item():.6f} | "
				  f"Data={components[0].item():.6f} | PDE={components[1].item():.6f} | "
				  f"BC={components[2].item():.6f} | IC={components[3].item():.6f}")

		if epoch % save_epochs == 0 and epoch != 0:
			# Save models
			torch.save(pinn.NN_sigma.state_dict(), f'{dirname}/NN_sigma_{epoch}.pth')
			torch.save(pinn.NN_price.state_dict(), f'{dirname}/NN_price_{epoch}.pth')
			
			# Plot surfaces
			_plot_surfaces(pinn, dirname, epoch, S_min, S_max, t_min, t_max, N_plot)
			
			# Plot losses
			_plot_losses(loss_history, loss_data_history, loss_pde_history, 
						loss_bc_history, loss_ic_history, dirname, epoch)

	# Final save
	torch.save(pinn.NN_sigma.state_dict(), f'{dirname}/NN_sigma_final.pth')
	torch.save(pinn.NN_price.state_dict(), f'{dirname}/NN_price_final.pth')
	_plot_surfaces(pinn, dirname, 'final', S_min, S_max, t_min, t_max, N_plot)
	_plot_losses(loss_history, loss_data_history, loss_pde_history, 
				loss_bc_history, loss_ic_history, dirname, 'final')

	print()
	print('Training completed!')
	print(f'Results saved in: {dirname}')

if __name__ == '__main__':
	main()

