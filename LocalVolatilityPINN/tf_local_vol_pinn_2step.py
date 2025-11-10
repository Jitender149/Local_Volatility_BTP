# ===============================================================
# Option Pricing and Local Volatility Surface using PINN
# Based on Bae, Kang, and Lee (2021)
# 
# Step 1: Train a neural network to approximate the price surface C(K, T)
# Step 2: Use automatic differentiation and Dupire's equation to compute σ_loc(K, T)
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
num_epochs = 30000
print_epochs = 2500
save_epochs = 10000

# Financial parameters
S0 = 1000.0
r = 0.04

# Strike & maturity domains (from project)
K_min, K_max = 500.0, 3000.0
T_min, T_max = 0.3, 1.5

# Learning rate
lr = 1e-3

# Loss weights (can be overridden via command-line arguments)
lambda_data = 1.0
lambda_pde = 1e-4
lambda_smooth = 1e-3

# MC Params - Small dataset (3×6)
N, m = 3, 6  # 3 maturities, 6 strikes each
M = 10**6
N_t = 150
dt = 0.01

# Plotting params
N_plot = 256
m_plot = 256

def parse_arguments():
	"""Parse command-line arguments for weight tuning"""
	parser = argparse.ArgumentParser(description='Local Volatility PINN 2-Step Training')
	parser.add_argument('--lambda_data', type=float, default=1.0, help='Data loss weight')
	parser.add_argument('--lambda_pde', type=float, default=1e-4, help='PDE residual weight')
	parser.add_argument('--lambda_smooth', type=float, default=1e-3, help='Smoothness regularization weight')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--num_epochs', type=int, default=30000, help='Number of training epochs')
	args = parser.parse_args()
	return args

# ---------------------------------------------------------------
# SYNTHETIC DATA GENERATION (Monte Carlo)
# ---------------------------------------------------------------

def exact_sigma_local(t, x):
	"""
	Define local volatility, which is a function of t and x = S/S0
	From project: y_ = sqrt(x + 0.1) * (t + 0.1), sigma_ = 0.3 + y_ * exp(-y_)
	"""
	y_ = tf.sqrt(x + 0.1) * (t + 0.1)
	sigma_ = 0.3 + y_ * tf.exp(-y_)
	return sigma_

class LoadData:
	def __init__(self):
		super(LoadData, self).__init__()
		self.N = N
		self.m = m
		self.N_t = N_t
		self.M = M

	def exact_sigma(self, t, x):
		"""Local volatility function"""
		return exact_sigma_local(t, x)

	def run_mc(self):
		"""Run Monte Carlo simulation"""
		M = self.M
		N_t = self.N_t
		dt = 0.01

		S_0 = tf.reshape(tf.constant([S0], dtype=data_type), [-1, 1])
		t_all = tf.cast(tf.reshape(np.linspace(0, N_t*dt, N_t), [-1, 1]), dtype=data_type)
		S_list = [tf.cast(tf.reshape(np.full(M, S_0[0]), [1, M]), dtype=data_type)]

		# Generate random increments
		dW_list = tf.cast(tf.concat([np.random.normal(0, 1, size=[N_t, 1]) * np.sqrt(dt) for i in range(M)], axis=1), dtype=data_type)

		for i in range(N_t-1):
			t_now = t_all[i]
			S_now = S_list[-1]
			S_new = S_now + r * S_now * dt + self.exact_sigma(t_now, S_now/S_0) * S_now * dW_list[i]
			S_list.append(S_new)

		S_matrix = tf.concat(S_list, axis=0)
		print(f'S_t obtained by solving local volatility SDE M = {M} times from t = [0, {N_t*dt}]')
		return S_matrix, t_all

	def get_T_K(self):
		"""Get T, K, and exact option prices from Monte Carlo"""
		N = self.N
		m = self.m
		N_t = self.N_t

		S_matrix, t_all = self.run_mc()

		# Select maturities
		indices = tf.cast(tf.linspace(30, N_t-1, N), tf.int32)
		t_all_T = tf.gather(t_all, indices)
		S_T = tf.gather(S_matrix, indices, axis=0)

		# Create strike-maturity grid
		T = tf.repeat(tf.reshape(t_all_T, [-1, 1]), m, axis=1)
		K = tf.cast(tf.repeat(tf.reshape(np.linspace(K_min, K_max, m), [1, -1]), len(T), axis=0), dtype=data_type)

		def exact_phi(T, K):
			"""Compute option price per maturity-strike pair using Monte Carlo"""
			E_ = tf.concat([tf.reshape(tf.reduce_mean(tf.nn.relu(tf.expand_dims(S_T[i], axis=0) - tf.expand_dims(K[i], axis=1)), axis=1), [1, -1]) for i in range(N)], axis=0)
			phi_ = tf.exp(-r * T) * E_
			return phi_

		phi_exact = exact_phi(T, K)
		T_nn = tf.reshape(T, [-1, 1])
		K_nn = tf.reshape(K, [-1, 1])
		phi_ref = tf.reshape(phi_exact, [-1, 1])

		return T_nn, K_nn, phi_ref, S_matrix, t_all

# ---------------------------------------------------------------
# STEP 1: PRICE SURFACE NETWORK
# ---------------------------------------------------------------

class PriceNetwork(tf.keras.Model):
	"""Neural network approximating the option price surface C(K, T)."""
	
	def __init__(self, K_min, K_max, T_min, T_max):
		super(PriceNetwork, self).__init__()
		self.K_min, self.K_max = K_min, K_max
		self.T_min, self.T_max = T_min, T_max
		
		# Build network
		inputs = tf.keras.Input(shape=(2,))
		x = inputs
		for h in [64, 64, 64]:
			x = layers.Dense(h, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
		outputs = layers.Dense(1)(x)
		self.net = Model(inputs, outputs, name="PriceNetwork")
	
	def _normalize(self, K, T):
		"""Normalize inputs to [-1, 1] for stable training."""
		K_norm = 2 * (K - self.K_min) / (self.K_max - self.K_min) - 1
		T_norm = 2 * (T - self.T_min) / (self.T_max - T_min) - 1
		return K_norm, T_norm
	
	def call(self, K, T):
		K_norm, T_norm = self._normalize(K, T)
		inputs = tf.concat([K_norm, T_norm], axis=1)
		return self.net(inputs)

# ---------------------------------------------------------------
# STEP 2: DUPIRE VOLATILITY COMPUTATION
# ---------------------------------------------------------------

def calculate_dupire_volatility(price_model, K, T, r):
	"""
	Computes σ_loc(K, T) using Dupire's local volatility formula:
	σ²(K,T) = [2 * (∂C/∂T + rK ∂C/∂K)] / [K² ∂²C/∂K²]
	"""
	with tf.GradientTape(persistent=True) as tape2:
		tape2.watch(K)
		with tf.GradientTape(persistent=True) as tape1:
			tape1.watch([K, T])
			C = price_model(K, T)
		C_t = tape1.gradient(C, T)
		C_k = tape1.gradient(C, K)
	C_kk = tape2.gradient(C_k, K)
	del tape1, tape2
	
	# Dupire's formula
	numerator = 2.0 * (C_t + r * K * C_k)
	denominator = tf.square(K) * (C_kk + 1e-8)
	local_variance = tf.maximum(numerator / (denominator + 1e-8), 0.0)
	sigma_loc = tf.sqrt(local_variance)
	return sigma_loc

# ---------------------------------------------------------------
# LOSS FUNCTIONS
# ---------------------------------------------------------------

def loss_data_fit(price_model, K_train, T_train, C_train):
	"""Data fitting loss: MSE on option prices"""
	C_pred = price_model(K_train, T_train)
	return tf.reduce_mean(tf.square(C_pred - C_train))

def loss_pde_residual(price_model, K_colloc, T_colloc, r):
	"""PDE residual loss: Black-Scholes PDE should be satisfied"""
	with tf.GradientTape(persistent=True) as tape2:
		tape2.watch(K_colloc)
		with tf.GradientTape(persistent=True) as tape1:
			tape1.watch([K_colloc, T_colloc])
			C = price_model(K_colloc, T_colloc)
		C_t = tape1.gradient(C, T_colloc)
		C_k = tape1.gradient(C, K_colloc)
	C_kk = tape2.gradient(C_k, K_colloc)
	del tape1, tape2
	
	# Compute local volatility from Dupire
	sigma_loc = calculate_dupire_volatility(price_model, K_colloc, T_colloc, r)
	
	# Black-Scholes PDE: ∂C/∂T + (1/2)σ²K²∂²C/∂K² + rK∂C/∂K - rC = 0
	pde_res = C_t + 0.5 * sigma_loc**2 * K_colloc**2 * C_kk + r * K_colloc * C_k - r * C
	return tf.reduce_mean(tf.square(pde_res))

def loss_smoothness(price_model, K_train, T_train):
	"""Smoothness regularization on price surface"""
	with tf.GradientTape(persistent=True) as tape:
		tape.watch([K_train, T_train])
		C = price_model(K_train, T_train)
	C_K = tape.gradient(C, K_train)
	C_T = tape.gradient(C, T_train)
	del tape
	smooth = tf.reduce_mean(C_K**2 + C_T**2)
	return smooth

# ---------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------

def _plot_surfaces(price_model, dirname, step, K_nn, T_nn, phi_ref, processdata):
	"""Plot learned price and volatility surfaces"""
	# Generate fine grid for plotting
	K_min_plot, K_max_plot = tf.reduce_min(K_nn), tf.reduce_max(K_nn)
	T_min_plot, T_max_plot = tf.reduce_min(T_nn), tf.reduce_max(T_nn)
	
	K_plot = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(K_min_plot, K_max_plot, N_plot), dtype=data_type), [-1, 1]), m_plot, axis=1), [-1, 1])
	T_plot = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(T_min_plot, T_max_plot, m_plot), dtype=data_type), [1, -1]), N_plot, axis=0), [-1, 1])
	
	# Predictions
	phi_pred_plot = price_model(K_plot, T_plot)
	sigma_pred_plot = calculate_dupire_volatility(price_model, K_plot, T_plot, r)
	
	# Exact local volatility for comparison
	# Note: We need to compute exact local vol from exact prices or use the exact_sigma function
	# For now, compute from exact_sigma function
	t_plot_norm = T_plot
	x_plot_norm = K_plot / S0
	sigma_exact_plot = exact_sigma_local(t_plot_norm, x_plot_norm)
	
	# Reshape for plotting
	K_plot_2d = tf.reshape(K_plot, [N_plot, m_plot])
	T_plot_2d = tf.reshape(T_plot, [N_plot, m_plot])
	phi_pred_2d = tf.reshape(phi_pred_plot, [N_plot, m_plot])
	phi_ref_2d = tf.reshape(phi_ref, [N, m])
	K_nn_2d = tf.reshape(K_nn, [N, m])
	T_nn_2d = tf.reshape(T_nn, [N, m])
	sigma_pred_2d = tf.reshape(sigma_pred_plot, [N_plot, m_plot])
	sigma_exact_2d = tf.reshape(sigma_exact_plot, [N_plot, m_plot])
	
	# Price surfaces
	fig = plt.figure(figsize=(24, 8), dpi=450)
	
	ax = fig.add_subplot(2, 3, 1, projection="3d")
	ax.plot_surface(K_plot_2d, T_plot_2d, phi_pred_2d, cmap=cm.coolwarm, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("Price C(K,T)")
	ax.set_title("Learned Option Price Surface")
	
	ax = fig.add_subplot(2, 3, 2, projection="3d")
	ax.plot_surface(K_nn_2d, T_nn_2d, phi_ref_2d, cmap=cm.coolwarm, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("Price C(K,T)")
	ax.set_title("Exact Option Price Surface (MC)")
	
	ax = fig.add_subplot(2, 3, 3, projection="3d")
	err_phi = phi_pred_2d - tf.reshape(price_model(K_nn, T_nn), [N, m])
	ax.plot_surface(K_nn_2d, T_nn_2d, err_phi, cmap=cm.RdBu_r, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("Error")
	ax.set_title("Price Error Surface")
	
	# Volatility surfaces
	ax = fig.add_subplot(2, 3, 4, projection="3d")
	ax.plot_surface(K_plot_2d, T_plot_2d, sigma_pred_2d, cmap=cm.viridis, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("σ_loc(K,T)")
	ax.set_title("Learned Local Volatility Surface (Dupire)")
	
	ax = fig.add_subplot(2, 3, 5, projection="3d")
	ax.plot_surface(K_plot_2d, T_plot_2d, sigma_exact_2d, cmap=cm.viridis, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("σ_loc(K,T)")
	ax.set_title("Exact Local Volatility Surface")
	
	ax = fig.add_subplot(2, 3, 6, projection="3d")
	err_sigma = sigma_pred_2d - sigma_exact_2d
	ax.plot_surface(K_plot_2d, T_plot_2d, err_sigma, cmap=cm.RdBu_r, linewidth=0)
	ax.set_xlabel("Strike (K)")
	ax.set_ylabel("Maturity (T)")
	ax.set_zlabel("Error")
	ax.set_title("Volatility Error Surface")
	
	plt.tight_layout()
	plt.savefig(os.path.join(dirname, f'surfaces_{step}.png'))
	if show:
		plt.show()
	else:
		plt.close()

def _plot_losses(loss_data_list, loss_pde_list, loss_smooth_list, dirname, step):
	"""Plot training loss curves"""
	fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=450)
	
	# Component losses
	ax[0].semilogy(loss_data_list, label='Data Loss', linewidth=1)
	ax[0].semilogy(loss_pde_list, label='PDE Loss', linewidth=1)
	ax[0].semilogy(loss_smooth_list, label='Smooth Loss', linewidth=1)
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel('Loss')
	ax[0].set_title('Component Losses')
	ax[0].legend()
	ax[0].grid(True, alpha=0.3)
	
	# Total loss
	total_loss_list = [d + lambda_pde * p + lambda_smooth * s for d, p, s in zip(loss_data_list, loss_pde_list, loss_smooth_list)]
	ax[1].semilogy(total_loss_list, label='Total Loss', linewidth=1)
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Loss')
	ax[1].set_title('Total Loss')
	ax[1].legend()
	ax[1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(dirname, f'losses_{step}.png'))
	if show:
		plt.show()
	else:
		plt.close()

# ---------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------

def main():
	# Parse command-line arguments
	args = parse_arguments()
	global lambda_data, lambda_pde, lambda_smooth, lr, num_epochs
	lambda_data = args.lambda_data
	lambda_pde = args.lambda_pde
	lambda_smooth = args.lambda_smooth
	lr = args.lr
	num_epochs = args.num_epochs

	print('Starting Local Volatility PINN 2-Step Training')
	print('')
	print(f'Loss weights: lambda_data={lambda_data}, lambda_pde={lambda_pde}, lambda_smooth={lambda_smooth}')
	print(f'Learning rate: {lr}, Epochs: {num_epochs}')
	print()

	# Create results directory
	identifier = f'local_vol_pinn_2step'
	folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
	dirname = folder_name_save
	os.makedirs(dirname, exist_ok=True)

	# Load data
	processdata = LoadData()
	T_nn, K_nn, phi_ref, S_matrix, t_all = processdata.get_T_K()
	
	# Prepare training data
	K_train = K_nn
	T_train = T_nn
	C_train = phi_ref
	
	print(f'Training data shape: K={K_train.shape}, T={T_train.shape}, C={C_train.shape}')
	print(f'K range: [{tf.reduce_min(K_nn):.2f}, {tf.reduce_max(K_nn):.2f}]')
	print(f'T range: [{tf.reduce_min(T_nn):.2f}, {tf.reduce_max(T_nn):.2f}]')
	print()

	# Initialize price network
	price_model = PriceNetwork(K_min, K_max, T_min, T_max)
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	
	# Collocation points for PDE residual
	N_colloc = 2000
	K_colloc = tf.random.uniform((N_colloc, 1), minval=K_min, maxval=K_max, dtype=data_type)
	T_colloc = tf.random.uniform((N_colloc, 1), minval=T_min, maxval=T_max, dtype=data_type)
	
	print(f'Collocation points: {N_colloc}')
	print()

	# Training loop
	loss_data_list = []
	loss_pde_list = []
	loss_smooth_list = []
	
	print("Training Step 1: Learning the Price Surface C(K, T)...")
	print("(Step 2: Local volatility computed via Dupire's equation during training)")
	print()
	
	for epoch in range(num_epochs + 1):
		with tf.GradientTape() as tape:
			# Data fitting loss
			L_data = loss_data_fit(price_model, K_train, T_train, C_train)
			
			# PDE residual loss
			L_pde = loss_pde_residual(price_model, K_colloc, T_colloc, r)
			
			# Smoothness regularization
			L_smooth = loss_smoothness(price_model, K_train, T_train)
			
			# Total loss
			total_loss = lambda_data * L_data + lambda_pde * L_pde + lambda_smooth * L_smooth
		
		# Backpropagation
		grads = tape.gradient(total_loss, price_model.trainable_variables)
		optimizer.apply_gradients(zip(grads, price_model.trainable_variables))
		
		# Store losses
		loss_data_list.append(float(L_data))
		loss_pde_list.append(float(L_pde))
		loss_smooth_list.append(float(L_smooth))
		
		if epoch % print_epochs == 0:
			rmse_fit = tf.sqrt(tf.reduce_mean(tf.square(price_model(K_nn, T_nn) - phi_ref)))
			print(f"Epoch {epoch:05d} | Total Loss = {total_loss:.6f} | "
				  f"Data={L_data:.6f} | PDE={L_pde:.6f} | Smooth={L_smooth:.6f} | RMSE={rmse_fit:.6f}")
		
		if epoch % save_epochs == 0 and epoch != 0:
			# Save model
			tf.keras.models.save_model(price_model.net, f'{dirname}/price_model_{epoch}.keras', overwrite=True)
			
			# Plot surfaces
			_plot_surfaces(price_model, dirname, epoch, K_nn, T_nn, phi_ref, processdata)
			
			# Plot losses
			_plot_losses(loss_data_list, loss_pde_list, loss_smooth_list, dirname, epoch)
	
	# Final save
	tf.keras.models.save_model(price_model.net, f'{dirname}/price_model_final.keras', overwrite=True)
	_plot_surfaces(price_model, dirname, 'final', K_nn, T_nn, phi_ref, processdata)
	_plot_losses(loss_data_list, loss_pde_list, loss_smooth_list, dirname, 'final')
	
	# Compute final local volatility surface
	print()
	print("Step 2: Computing final local volatility surface via Dupire's equation...")
	K_final = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(K_min, K_max, N_plot), dtype=data_type), [-1, 1]), m_plot, axis=1), [-1, 1])
	T_final = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(T_min, T_max, m_plot), dtype=data_type), [1, -1]), N_plot, axis=0), [-1, 1])
	sigma_final = calculate_dupire_volatility(price_model, K_final, T_final, r)
	
	print(f'Final local volatility range: [{tf.reduce_min(sigma_final):.4f}, {tf.reduce_max(sigma_final):.4f}]')
	print()
	print('Training completed!')
	print(f'Results saved in: {dirname}')

if __name__ == '__main__':
	main()

