# ===============================================================
# Local Volatility PINN based on Bae, Kang & Lee (2021)
# ===============================================================

import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import datetime
import logging
import argparse

tf.random.set_seed(42)

VariableSpec = resource_variable_ops.VariableSpec

# Device setup
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if len(gpus) != 0:
	device = gpus[0]
	tf.config.set_visible_devices(device, 'GPU')
	tf.config.experimental.set_memory_growth(device, True)
else:
	device = cpus[0]
	tf.config.set_visible_devices(device, 'CPU')

print('Simulation is running on:')
print(f'device = {device}')
print()

data_type = tf.float32
tf.keras.backend.set_floatx('float32')

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
	parser = argparse.ArgumentParser(description='Local Volatility PINN Training')
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
	return 0.3 + 0.2 * tf.sin(np.pi * S) * tf.exp(-t)

def exact_price(S, t):
	"""Dummy target price surface (e.g., precomputed option prices)"""
	# Just for training demonstration; in real setting use Monte Carlo or market data
	sigma = exact_sigma_local(S, t)
	return tf.exp(-0.5 * sigma * S) * tf.maximum(S - 1.0, 0)

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

	print('Starting Local Volatility PINN Training')
	print('')
	print(f'Loss weights: lambda_data={lambda_data}, lambda_pde={lambda_pde}, lambda_bc={lambda_bc}, lambda_ic={lambda_ic}')
	print(f'Learning rate: {lr}, Epochs: {num_epochs}')
	print()

	# Create results directory
	identifier = f'local_vol_pinn'
	folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
	dirname = folder_name_save
	os.makedirs(dirname, exist_ok=True)

	# Create collocation & data points
	print('Generating training data...')
	S_data = tf.random.uniform((N_data, 1), S_min, S_max, dtype=data_type)
	t_data = tf.random.uniform((N_data, 1), t_min, t_max, dtype=data_type)
	V_data = exact_price(S_data, t_data)

	# Boundary / initial conditions
	# Initial condition: at maturity t=T (normalized t=1) → payoff
	S_ic = tf.linspace(S_min, S_max, N_ic)[:, None]
	t_ic = tf.ones_like(S_ic) * t_max
	V_ic = tf.maximum(S_ic - 1.0, 0)  # (S - K)+ payoff, K=1 normalized

	# Lower boundary: S=0 → V=0
	S_bc0 = tf.zeros((N_bc, 1), dtype=data_type)
	t_bc0 = tf.linspace(t_min, t_max, N_bc)[:, None]
	V_bc0 = tf.zeros_like(S_bc0)

	# Upper boundary: linear asymptote V ≈ S - e^{-r(T-t)}K
	S_bc1 = tf.ones((N_bc, 1), dtype=data_type) * S_max
	t_bc1 = tf.linspace(t_min, t_max, N_bc)[:, None]
	V_bc1 = S_bc1 - tf.exp(-r * (1 - t_bc1))

	# Collocation points (for PDE residual)
	S_colloc = tf.random.uniform((N_colloc, 1), S_min, S_max, dtype=data_type)
	t_colloc = tf.random.uniform((N_colloc, 1), t_min, t_max, dtype=data_type)

	print(f'Data points: {N_data}')
	print(f'Collocation points: {N_colloc}')
	print(f'Boundary condition points: {N_bc * 2}')
	print(f'Initial condition points: {N_ic}')
	print()

	# ---------------------------------------------------------------
	# NEURAL NETWORKS: SigmaNet and PriceNet
	# ---------------------------------------------------------------

	def build_sigma_net():
		inputs = tf.keras.Input(shape=(2,))
		x = inputs
		for h in [64, 64, 64]:
			x = layers.Dense(h, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
		sigma_out = layers.Dense(1, activation='softplus')(x)
		return Model(inputs, sigma_out, name="SigmaNet")

	def build_price_net():
		inputs = tf.keras.Input(shape=(2,))
		x = inputs
		for h in [64, 64, 64]:
			x = layers.Dense(h, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
		V_out = layers.Dense(1, activation=None)(x)
		return Model(inputs, V_out, name="PriceNet")

	# ---------------------------------------------------------------
	# PHYSICS-INFORMED MODEL
	# ---------------------------------------------------------------

	class LocalVolPINN(tf.keras.Model):
		def __init__(self, r):
			super(LocalVolPINN, self).__init__()
			self.r = r
			self.NN_sigma = build_sigma_net()
			self.NN_price = build_price_net()
			self.optimizer = tf.keras.optimizers.Adam(lr)

		# ------------------------
		# PDE residual (Dupire/Black-Scholes)
		# ------------------------
		def loss_pde(self, S, t):
			with tf.GradientTape(persistent=True) as tape:
				tape.watch([S, t])
				inp = tf.concat([S, t], axis=1)
				V = self.NN_price(inp)
				sigma = self.NN_sigma(inp)
			V_t = tape.gradient(V, t)
			V_S = tape.gradient(V, S)
			V_SS = tape.gradient(V_S, S)
			del tape

			# Black-Scholes PDE: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
			pde_res = V_t + 0.5 * sigma**2 * S**2 * V_SS + self.r * S * V_S - self.r * V
			return tf.reduce_mean(tf.square(pde_res))

		# ------------------------
		# Total loss
		# ------------------------
		def total_loss(self, data, bc, ic):
			(S_d, t_d, V_d) = data
			(S_bc0, t_bc0, V_bc0, S_bc1, t_bc1, V_bc1) = bc
			(S_ic, t_ic, V_ic) = ic

			# Data loss
			L_data = tf.reduce_mean(tf.square(self.NN_price(tf.concat([S_d, t_d], 1)) - V_d))

			# PDE residual
			L_pde = self.loss_pde(S_colloc, t_colloc)

			# Boundary losses
			L_bc0 = tf.reduce_mean(tf.square(self.NN_price(tf.concat([S_bc0, t_bc0], 1)) - V_bc0))
			L_bc1 = tf.reduce_mean(tf.square(self.NN_price(tf.concat([S_bc1, t_bc1], 1)) - V_bc1))

			# Initial (terminal payoff) loss
			L_ic = tf.reduce_mean(tf.square(self.NN_price(tf.concat([S_ic, t_ic], 1)) - V_ic))

			total = (
				lambda_data * L_data +
				lambda_pde * L_pde +
				lambda_bc * (L_bc0 + L_bc1) +
				lambda_ic * L_ic
			)

			return total, (L_data, L_pde, L_bc0 + L_bc1, L_ic)

		# ------------------------
		# Training step
		# ------------------------
		@tf.function
		def train_step(self, data, bc, ic):
			with tf.GradientTape() as tape:
				total_loss, components = self.total_loss(data, bc, ic)
			grads = tape.gradient(total_loss, self.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
			return total_loss, components

		# ------------------------
		# Save models
		# ------------------------
		def save_models(self, dirname, iter_=None):
			if iter_ is not None:
				tf.keras.models.save_model(self.NN_sigma, f'{dirname}/NN_sigma_{iter_}.keras', overwrite=True)
				tf.keras.models.save_model(self.NN_price, f'{dirname}/NN_price_{iter_}.keras', overwrite=True)
			else:
				tf.keras.models.save_model(self.NN_sigma, f'{dirname}/NN_sigma_final.keras', overwrite=True)
				tf.keras.models.save_model(self.NN_price, f'{dirname}/NN_price_final.keras', overwrite=True)

	# ---------------------------------------------------------------
	# TRAINING LOOP
	# ---------------------------------------------------------------

	pinn = LocalVolPINN(r)
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
		total_loss, components = pinn.train_step(data, bc, ic)

		loss_history.append(float(total_loss))
		loss_data_history.append(float(components[0]))
		loss_pde_history.append(float(components[1]))
		loss_bc_history.append(float(components[2]))
		loss_ic_history.append(float(components[3]))

		if epoch % print_epochs == 0:
			print(f"Epoch {epoch:05d} | Total Loss = {total_loss:.6f} | "
				  f"Data={components[0]:.6f} | PDE={components[1]:.6f} | "
				  f"BC={components[2]:.6f} | IC={components[3]:.6f}")

		if epoch % save_epochs == 0 and epoch != 0:
			# Save models
			pinn.save_models(dirname, epoch)
			
			# Plot surfaces
			_plot_surfaces(pinn, dirname, epoch, S_min, S_max, t_min, t_max, N_plot)
			
			# Plot losses
			_plot_losses(loss_history, loss_data_history, loss_pde_history, 
						loss_bc_history, loss_ic_history, dirname, epoch)

	# Final save
	pinn.save_models(dirname, 'final')
	_plot_surfaces(pinn, dirname, 'final', S_min, S_max, t_min, t_max, N_plot)
	_plot_losses(loss_history, loss_data_history, loss_pde_history, 
				loss_bc_history, loss_ic_history, dirname, 'final')

	print()
	print('Training completed!')
	print(f'Results saved in: {dirname}')

# ---------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------

def _plot_surfaces(pinn, dirname, step, S_min, S_max, t_min, t_max, N_plot):
	"""Plot learned price and volatility surfaces"""
	S_grid = np.linspace(S_min, S_max, N_plot)
	t_grid = np.linspace(t_min, t_max, N_plot)
	S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)
	inp = np.stack([S_mesh.flatten(), t_mesh.flatten()], axis=1)
	
	V_pred = pinn.NN_price(inp).numpy().reshape(N_plot, N_plot)
	sigma_pred = pinn.NN_sigma(inp).numpy().reshape(N_plot, N_plot)
	
	# Exact surfaces for comparison
	V_exact = exact_price(tf.constant(S_mesh.flatten()[:, None], dtype=data_type),
						  tf.constant(t_mesh.flatten()[:, None], dtype=data_type)).numpy().reshape(N_plot, N_plot)
	sigma_exact = exact_sigma_local(tf.constant(S_mesh.flatten()[:, None], dtype=data_type),
									tf.constant(t_mesh.flatten()[:, None], dtype=data_type)).numpy().reshape(N_plot, N_plot)
	
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

if __name__ == '__main__':
	main()

