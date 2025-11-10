"""
Sigma-Price Network: Direct Learning of Local Volatility

This script implements a two-network approach:
1. Sigma Network: Learns σ_local(S, K, τ) directly
2. Price Network: Uses σ_local to predict option prices

Uses data generation from Synthetic_Data_Tensorflow (Monte Carlo simulation)
"""

import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
from tensorflow.keras import layers, Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import logging, os
import datetime
import math
import argparse

tf.random.set_seed(42)

VariableSpec = resource_variable_ops.VariableSpec

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
data_type_nn = tf.float32
tf.keras.backend.set_floatx('float32')

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

show = False
num_epochs = 30000
print_epochs = 2500
save_epochs = 10000

# NN Params
hidden_dim_sigma = 20000  # Large hidden dimension like PyTorch code
hidden_dim_price = 20000
# Default lambda values
lambda_smooth = 1e-3
lambda_pde = 1e-4
use_adaptive_weights = True
adaptive_update_freq = 1000
lr = 10**-3

# Plotting params
N_ = 256
m_ = 256

# MC Params - Small dataset (3×6)
N, m = 3, 6  # 3 maturities, 6 strikes each
S0 = 1000
r_ = 0.04
M = 10**4
N_t = 1000
dt = 10**-3

def parse_arguments():
	"""Parse command-line arguments for weight tuning"""
	parser = argparse.ArgumentParser(description='Sigma-Price Network Training')
	parser.add_argument('--lambda_smooth', type=float, default=1e-3, help='Smoothness regularization weight')
	parser.add_argument('--lambda_pde', type=float, default=1e-4, help='PDE residual weight')
	parser.add_argument('--no_adaptive', action='store_true', help='Disable adaptive weight balancing')
	parser.add_argument('--adaptive_freq', type=int, default=1000, help='Frequency for adaptive weight updates')
	args = parser.parse_args()
	return args

def main():
	# Parse command-line arguments
	args = parse_arguments()
	global lambda_smooth, lambda_pde, use_adaptive_weights, adaptive_update_freq
	lambda_smooth = args.lambda_smooth
	lambda_pde = args.lambda_pde
	use_adaptive_weights = not args.no_adaptive
	adaptive_update_freq = args.adaptive_freq

	print('Starting Sigma-Price Network Training')
	print('')

	identifier = f'sigma_price_network_small'

	folder_name_load = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
	folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
	dirname = folder_name_save
	os.makedirs(dirname, exist_ok=True)

	S_0 = tf.constant(S0, dtype=data_type)
	r = tf.constant(r_, dtype=data_type)

	class LoadData:
		def __init__(self):
			super(LoadData, self).__init__()
			self.N = N
			self.m = m
			self.N_t = 150
			self.M = 10**6

		def save_nn(self, NN_sigma, NN_price, iter_=None):
			tf.keras.models.save_model(NN_sigma,
									   filepath = f'{folder_name_save}/NN_sigma_{iter_}.keras',
									   overwrite = True)
			tf.keras.models.save_model(NN_price,
									   filepath = f'{folder_name_save}/NN_price_{iter_}.keras',
									   overwrite = True)

		def load_nn(self, folder_name_load, iter_=None):
			NN_sigma = tf.keras.models.load_model(f'{folder_name_load}/NN_sigma_{iter_}.keras')
			NN_price = tf.keras.models.load_model(f'{folder_name_load}/NN_price_{iter_}.keras')
			return NN_sigma, NN_price

		def exact_sigma(self, t, x):
			"""define local volatility, which is a function of t and x"""
			y_ = tf.sqrt(x + 0.1) * (t + 0.1)
			sigma_ = 0.3 + y_ * tf.exp(-y_)
			return sigma_

		def run_mc(self):
			"""Run Monte Carlo simulation"""
			d = 1
			M = self.M
			N_t = self.N_t
			dt = 0.01

			S_0 = tf.reshape(tf.constant([1000.0], dtype=data_type), [-1, 1])
			t_all = tf.cast(tf.reshape(np.linspace(0, N_t*dt, N_t), [-1, 1]), dtype=data_type)
			S_list = [tf.cast(tf.reshape(np.full(M, S_0[0]), [1, M]), dtype=data_type)]

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
			"""Get T, K, and exact option prices"""
			N = self.N
			m = self.m
			N_t = self.N_t

			S_matrix, t_all = self.run_mc()

			indices = tf.cast(tf.linspace(30, N_t-1, N), tf.int32)
			t_all_T = tf.gather(t_all, indices)
			S_T = tf.gather(S_matrix, indices, axis=0)

			T = tf.repeat(tf.reshape(t_all_T, [-1, 1]), m, axis=1)
			K = tf.cast(tf.repeat(tf.reshape(np.linspace(500, 3000, m), [1, -1]), len(T), axis=0), dtype=data_type)

			def exact_phi(T, K):
				"""compute option price per maturity-strike pair"""
				E_ = tf.concat([tf.reshape(tf.reduce_mean(tf.nn.relu(tf.expand_dims(S_T[i], axis=0) - tf.expand_dims(K[i], axis=1)), axis=1), [1, -1]) for i in range(N)], axis=0)
				phi_ = tf.exp(-r * T) * E_
				return phi_

			phi_exact = exact_phi(T, K)
			T_nn = tf.reshape(T, [-1, 1])
			K_nn = tf.reshape(K, [-1, 1])
			phi_ref = tf.reshape(phi_exact, [-1, 1])

			return T_nn, K_nn, phi_ref, S_matrix, t_all

	processdata = LoadData()

	T_nn, K_nn, phi_ref, S_matrix, t_all = processdata.get_T_K()

	# Prepare training data
	# For Sigma Network: input is (S, K, tau) where S is spot, K is strike, tau is time to maturity
	# For Price Network: input is (S, K, tau, sigma_local)
	S_train = tf.ones_like(K_nn) * S0  # Spot price
	KT_train = tf.concat([S_train, K_nn, T_nn], axis=1)  # (S, K, T)
	P_train = phi_ref

	print(f'Training data shape: SKT={KT_train.shape}, P={P_train.shape}')
	print(f'S range: [{tf.reduce_min(S_train):.2f}, {tf.reduce_max(S_train):.2f}]')
	print(f'K range: [{tf.reduce_min(K_nn):.2f}, {tf.reduce_max(K_nn):.2f}]')
	print(f'T range: [{tf.reduce_min(T_nn):.2f}, {tf.reduce_max(T_nn):.2f}]')
	print()

	class PhysicsModel(tf.keras.Model):
		def __init__(self, lambda_smooth=1e-3, lambda_pde=1e-4):
			super(PhysicsModel, self).__init__()
			self.lambda_smooth = lambda_smooth
			self.lambda_pde = lambda_pde
			self.S0 = tf.constant(S0, dtype=data_type)
			self.r = tf.constant(r_, dtype=data_type)

		# ---------------------------
		# Sigma Network: σ_local(K, τ)
		# ---------------------------
		def build_sigma_net(self, hidden_dim=20000):
			"""Sigma Network - outputs local volatility σ_local(K, τ)"""
			inputs = tf.keras.Input(shape=(2,))  # (K, τ)
			x = inputs
			# Single large hidden layer with softplus activation
			h = layers.Dense(hidden_dim, activation='softplus', 
							kernel_regularizer=regularizers.l2(1e-5))(x)
			# Output: local volatility (ensure positive)
			sigma_out = layers.Dense(1, activation='softplus')(h)
			model = Model(inputs, sigma_out, name="SigmaNet")
			return model

		# ---------------------------
		# Price Network: V(S, K, τ, σ_local)
		# ---------------------------
		def build_price_net(self, hidden_dim=20000):
			"""Price Network - outputs option price V(S, K, τ, σ_local)"""
			inputs = tf.keras.Input(shape=(4,))  # (S, K, τ, σ_local)
			x = inputs
			# Single large hidden layer with softplus activation
			h = layers.Dense(hidden_dim, activation='softplus',
							kernel_regularizer=regularizers.l2(1e-5))(x)
			# Output: option price (no activation, can be negative for deep OTM)
			price_out = layers.Dense(1, activation=None)(h)
			model = Model(inputs, price_out, name="PriceNet")
			return model

		def build_models(self):
			self.NN_sigma = self.build_sigma_net(hidden_dim=hidden_dim_sigma)
			self.NN_price = self.build_price_net(hidden_dim=hidden_dim_price)
			self.optimizer_NN_sigma = tf.keras.optimizers.Adam(learning_rate=lr)
			self.optimizer_NN_price = tf.keras.optimizers.Adam(learning_rate=lr)
			# Optimizers for adaptive loss weights
			self.optimizer_lambda_smooth = tf.keras.optimizers.Adam(learning_rate=1e-3)
			self.optimizer_lambda_pde = tf.keras.optimizers.Adam(learning_rate=1e-3)

		# ---------------------------
		# Forward pass helpers
		# ---------------------------
		def neural_sigma(self, K, tau):
			"""Return local volatility σ_local(K, τ)"""
			# Stack inputs: (K, τ)
			inp = tf.concat([tf.reshape(K, [-1, 1]), 
							tf.reshape(tau, [-1, 1])], axis=1)
			sigma_local = self.NN_sigma(inp)
			return tf.squeeze(sigma_local)

		def neural_price(self, S, K, tau, sigma_local):
			"""Return option price V(S, K, τ, σ_local)"""
			# Stack inputs: (S, K, τ, σ_local)
			inp = tf.concat([tf.reshape(S, [-1, 1]),
							tf.reshape(K, [-1, 1]),
							tf.reshape(tau, [-1, 1]),
							tf.reshape(sigma_local, [-1, 1])], axis=1)
			price = self.NN_price(inp)
			return tf.squeeze(price)

		def combined_forward(self, S, K, tau):
			"""Combined forward pass: sigma -> price"""
			sigma_local = self.neural_sigma(K, tau)
			price = self.neural_price(S, K, tau, sigma_local)
			return price, sigma_local

		# ---------------------------
		# Loss terms
		# ---------------------------
		def loss_price_mse(self):
			"""Data loss: MSE between predicted and exact prices"""
			S_vals = KT_train[:, 0:1]
			K_vals = KT_train[:, 1:2]
			T_vals = KT_train[:, 2:3]
			
			price_pred, _ = self.combined_forward(S_vals, K_vals, T_vals)
			return tf.reduce_mean(tf.square(price_pred - tf.squeeze(P_train)))

		def loss_smoothness(self):
			"""Smoothness regularizer on σ_local"""
			K_vals = KT_train[:, 1:2]
			T_vals = KT_train[:, 2:3]
			
			with tf.GradientTape(persistent=True) as tape:
				tape.watch([K_vals, T_vals])
				sigma = self.neural_sigma(K_vals, T_vals)
			
			# Compute gradients
			sigma_K = tape.gradient(sigma, K_vals)
			sigma_T = tape.gradient(sigma, T_vals)
			
			# Smoothness: minimize squared gradients
			smooth = tf.reduce_mean(sigma_K**2 + sigma_T**2)
			return smooth

		def loss_pde(self, n_colloc=200):
			"""PDE residual on random (S, K, T) grid"""
			# Sample collocation points within training data range
			S_min = tf.reduce_min(KT_train[:, 0:1])
			S_max = tf.reduce_max(KT_train[:, 0:1])
			K_min = tf.reduce_min(KT_train[:, 1:2])
			K_max = tf.reduce_max(KT_train[:, 1:2])
			T_min = tf.reduce_min(KT_train[:, 2:3])
			T_max = tf.reduce_max(KT_train[:, 2:3])
			
			Sc = tf.random.uniform([n_colloc, 1], minval=S_min, maxval=S_max, dtype=data_type)
			Kc = tf.random.uniform([n_colloc, 1], minval=K_min, maxval=K_max, dtype=data_type)
			Tc = tf.random.uniform([n_colloc, 1], minval=T_min, maxval=T_max, dtype=data_type)
			
			# Use nested GradientTapes for second derivatives (like original code)
			with tf.GradientTape(persistent=True) as tape2:
				tape2.watch(Sc)
				with tf.GradientTape(persistent=True) as tape1:
					tape1.watch([Sc, Kc, Tc])
					V, sigma_local = self.combined_forward(Sc, Kc, Tc)
				
				# First derivatives
				V_S = tape1.gradient(V, Sc)
				V_T = tape1.gradient(V, Tc)
			
			# Second derivative
			V_SS = tape2.gradient(V_S, Sc)
			
			# Black-Scholes PDE with local volatility:
			# ∂V/∂T + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V = 0
			r_val = self.r
			residual = V_T + 0.5 * (sigma_local**2) * (Sc**2) * V_SS + r_val * Sc * V_S - r_val * V
			
			return tf.reduce_mean(tf.square(residual))

		@tf.function
		def train_step(self, lambda_smooth_var, lambda_pde_var):
			with tf.GradientTape(persistent=True) as tape:
				loss_price = self.loss_price_mse()
				loss_pde = self.loss_pde()
				loss_smooth = self.loss_smoothness()
				total_loss = loss_price + lambda_pde_var * loss_pde + lambda_smooth_var * loss_smooth
			
			# Gradients for both networks
			grads_sigma = tape.gradient(total_loss, self.NN_sigma.trainable_variables)
			grads_price = tape.gradient(loss_price, self.NN_price.trainable_variables)
			
			# Apply gradients
			self.optimizer_NN_sigma.apply_gradients(zip(grads_sigma, self.NN_sigma.trainable_variables))
			self.optimizer_NN_price.apply_gradients(zip(grads_price, self.NN_price.trainable_variables))
			
			return loss_price, loss_pde, loss_smooth, total_loss
		
		def update_lambda_weights(self, lambda_smooth_var, lambda_pde_var, loss_price, loss_smooth, loss_pde):
			"""Update lambda weights using optimizers based on loss magnitudes"""
			# Compute gradients for lambda weights
			# We want to balance the contributions: lambda_i * loss_i ≈ loss_price
			with tf.GradientTape() as tape:
				tape.watch([lambda_smooth_var, lambda_pde_var])
				# Loss for lambda optimization: encourage balanced contributions
				# Target: minimize the difference between weighted losses and data loss
				lambda_loss = tf.square(lambda_smooth_var * loss_smooth - loss_price) + \
							  tf.square(lambda_pde_var * loss_pde - loss_price)
			
			# Gradients for lambda weights
			grads_lambda = tape.gradient(lambda_loss, [lambda_smooth_var, lambda_pde_var])
			
			# Apply gradients using optimizers with clipping to keep weights positive and reasonable
			if grads_lambda[0] is not None:
				# Update lambda_smooth using optimizer
				self.optimizer_lambda_smooth.apply_gradients([(grads_lambda[0], lambda_smooth_var)])
				# Clip to reasonable range
				lambda_smooth_var.assign(tf.clip_by_value(lambda_smooth_var, 1e-6, 1e3))
			
			if grads_lambda[1] is not None:
				# Update lambda_pde using optimizer
				self.optimizer_lambda_pde.apply_gradients([(grads_lambda[1], lambda_pde_var)])
				# Clip to reasonable range
				lambda_pde_var.assign(tf.clip_by_value(lambda_pde_var, 1e-6, 1e3))
			
			return lambda_smooth_var, lambda_pde_var

	physics = PhysicsModel(lambda_smooth=lambda_smooth, lambda_pde=lambda_pde)

	physics.build_models()

	class Plotter:
		def __init__(self):
			super(Plotter, self).__init__()

		def plot_res(self, loss_price_list, loss_smooth_list, loss_pde_list, error_sigma_list, step):
			# Generate grid for plotting
			T_min, T_max = tf.reduce_min(T_nn), tf.reduce_max(T_nn)
			K_min, K_max = tf.reduce_min(K_nn), tf.reduce_max(K_nn)

			# Create mesh grid
			T_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(T_min, T_max, N_), dtype=data_type), [-1, 1]), m_, axis=1), [-1, 1])
			K_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(K_min, K_max, m_), dtype=data_type), [1, -1]), N_, axis=0), [-1, 1])
			S_ = tf.ones_like(K_) * S0

			# Get predictions
			phi_pred_, sigma_nn_ = physics.combined_forward(S_, K_, T_)

			# Compute exact sigma for comparison
			# exact_sigma expects (t, x) where x = S/S0 (normalized spot price)
			t_exact = T_
			x_exact = S_ / S0  # Use S_ (spot price) not K_ (strike)
			sigma_exact_ = processdata.exact_sigma(t_exact, x_exact)

			err_surf_phi = (tf.reshape(phi_ref, [N, m]) - tf.reshape(physics.combined_forward(S_train, K_nn, T_nn)[0], [N, m]))
			err_surf_sigma = (tf.reshape(sigma_exact_, [N_, m_]) - tf.reshape(sigma_nn_, [N_, m_]))

			# Generate plots
			self._plot_neural_option_price(T_, K_, phi_pred_, err_surf_phi, step)
			self._plot_neural_local_volatility(T_, K_, sigma_nn_, sigma_exact_, err_surf_sigma, step)
			self._plot_losses(loss_price_list, loss_smooth_list, loss_pde_list, error_sigma_list, step)

		def _plot_neural_option_price(self, T_nn_, K_nn_, phi_pred_, err_surf_phi, step):
			fig = plt.figure(figsize=[24, 8], dpi=450)

			ax_1 = fig.add_subplot(1,3,1, projection='3d')
			ax_1.plot_surface(tf.reshape(K_nn_, [N_, m_]), tf.reshape(T_nn_, [N_, m_]), tf.reshape(phi_pred_, [N_, m_]), cmap=cm.RdBu_r, linewidth=0)
			ax_1.set_ylabel('Maturity: T')
			ax_1.set_xlabel('Strike price: K')
			ax_1.set_zlabel('Neural option price: phi')

			ax_2 = fig.add_subplot(1,3,2, projection='3d')
			ax_2.plot_surface(tf.reshape(K_nn, [N, m]), tf.reshape(T_nn, [N, m]), tf.reshape(phi_ref, [N, m]), cmap=cm.RdBu_r, linewidth=0)
			ax_2.set_ylabel('Maturity: T')
			ax_2.set_xlabel('Strike price: K')
			ax_2.set_zlabel('Exact option price: phi')

			ax_3 = fig.add_subplot(1,3,3, projection='3d')
			ax_3.plot_surface(tf.reshape(K_nn, [N, m]), tf.reshape(T_nn, [N, m]), tf.reshape(err_surf_phi, [N, m]), cmap=cm.RdBu_r, linewidth=0)
			ax_3.set_ylabel('Maturity: T')
			ax_3.set_xlabel('Strike price: K')
			ax_3.set_zlabel('Error phi')

			plt.savefig(os.path.join(dirname, f'phi_{step}.png'))
			if show:
				plt.show()
			else:
				plt.close()

		def _plot_neural_local_volatility(self, T_nn_, K_nn_, sigma_nn_, sigma_exact_, err_surf_sigma, step):
			fig = plt.figure(figsize=[24, 8], dpi=450)

			ax_1 = fig.add_subplot(1,3,1, projection='3d')
			ax_1.plot_surface(tf.reshape(K_nn_, [N_, m_]), tf.reshape(T_nn_, [N_, m_]), tf.reshape(sigma_nn_, [N_, m_]), cmap=cm.RdBu_r, linewidth=0)
			ax_1.set_ylabel('Maturity: T')
			ax_1.set_xlabel('Strike price: K')
			ax_1.set_zlabel('Neural local volatility: sigma')

			ax_2 = fig.add_subplot(1,3,2, projection='3d')
			ax_2.plot_surface(tf.reshape(K_nn_, [N_, m_]), tf.reshape(T_nn_, [N_, m_]), tf.reshape(sigma_exact_, [N_, m_]), cmap=cm.RdBu_r, linewidth=0)
			ax_2.set_ylabel('Maturity: T')
			ax_2.set_xlabel('Strike price: K')
			ax_2.set_zlabel('Exact local volatility: sigma')

			ax_3 = fig.add_subplot(1,3,3, projection='3d')
			ax_3.plot_surface(tf.reshape(K_nn_, [N_, m_]), tf.reshape(T_nn_, [N_, m_]), tf.reshape(err_surf_sigma, [N_, m_]), cmap=cm.RdBu_r, linewidth=0)
			ax_3.set_ylabel('Maturity: T')
			ax_3.set_xlabel('Strike price: K')
			ax_3.set_zlabel('Error sigma')

			plt.savefig(os.path.join(dirname, f'sigma_error_{step}.png'))
			if show:
				plt.show()
			plt.close()

		def _plot_losses(self, loss_price_list, loss_smooth_list, loss_pde_list, error_sigma_list, step):
			fig, ax = plt.subplots(1, 2, figsize=[12, 2], dpi=450)

			ax[0].semilogy(loss_price_list, label='loss_price')
			ax[0].semilogy(loss_smooth_list, label='loss_smooth')
			if len(loss_pde_list) > 0:
				ax[0].semilogy(loss_pde_list, label='loss_pde')
			ax[0].legend(loc='upper right')

			ax[1].plot(error_sigma_list, label='relative error sigma')
			ax[1].legend(loc='upper right')

			plt.savefig(os.path.join(dirname, f'losses_{step}.png'))
			if show:
				plt.show()
			plt.close()

	plotter = Plotter()

	class Trainer:
		def __init__(self):
			super(Trainer, self).__init__()

		def test(self):
			plotter.plot_res([], [], [], [], step=-1)

			time_0 = time.time()
			loss_price = physics.loss_price_mse()
			time_1 = time.time()
			print(f'loss_price = {loss_price}, computation time = {time_1 - time_0}')

			time_0 = time.time()
			loss_smooth = physics.loss_smoothness()
			time_1 = time.time()
			print(f'loss_smooth = {loss_smooth}, computation time = {time_1 - time_0}')

			time_0 = time.time()
			loss_pde = physics.loss_pde()
			time_1 = time.time()
			print(f'loss_pde = {loss_pde}, computation time = {time_1 - time_0}')

			physics.optimizer_NN_sigma.learning_rate.assign(10**-4)
			physics.optimizer_NN_price.learning_rate.assign(10**-4)

			lambda_smooth_var = tf.Variable(1e-3, dtype=data_type, trainable=True)
			lambda_pde_var = tf.Variable(1e-4, dtype=data_type, trainable=True)

			time_0 = time.time()
			loss_price, loss_pde, loss_smooth, total_loss = physics.train_step(lambda_smooth_var, lambda_pde_var)
			time_1 = time.time()

			print(f'computation time = {time_1 - time_0}')
			print(f'loss_price = {loss_price}, loss_smooth = {loss_smooth}, loss_pde = {loss_pde}, total_loss = {total_loss}')
			print()

		def run(self):
			learning_rate_ = lr
			physics.optimizer_NN_sigma.learning_rate.assign(learning_rate_)
			physics.optimizer_NN_price.learning_rate.assign(learning_rate_)

			loss_price_list = []
			loss_smooth_list = []
			loss_pde_list = []
			error_sigma_list = []
			rmse_sigma_list = []

			# Initialize lambda values as trainable variables for optimizer-based updates
			lambda_smooth_val = tf.Variable(lambda_smooth, dtype=data_type, trainable=True)
			lambda_pde_val = tf.Variable(lambda_pde, dtype=data_type, trainable=True)

			print(f'Starting training with weights: lambda_smooth={lambda_smooth_val.numpy():.6f}, lambda_pde={lambda_pde_val.numpy():.6f}')
			print(f'Using optimizer-based adaptive weight balancing (update frequency: {adaptive_update_freq} iterations)')
			print()

			for iter_ in range(num_epochs+1):
				loss_price, loss_pde, loss_smooth, total_loss = physics.train_step(lambda_smooth_val, lambda_pde_val)

				loss_price_list.append(loss_price)
				loss_smooth_list.append(loss_smooth)
				loss_pde_list.append(loss_pde)

				# Update lambda weights using optimizers
				if use_adaptive_weights and iter_ > 0 and iter_ % adaptive_update_freq == 0:
					lambda_smooth_val, lambda_pde_val = physics.update_lambda_weights(
						lambda_smooth_val, lambda_pde_val, loss_price, loss_smooth, loss_pde
					)
					
					print(f'[Iter {iter_}] Optimizer-based weight update:')
					print(f'  Losses: price={loss_price:.6f}, smooth={loss_smooth:.6f}, pde={loss_pde:.6f}')
					print(f'  Updated weights: lambda_smooth={lambda_smooth_val.numpy():.6f}, lambda_pde={lambda_pde_val.numpy():.6f}')
					print(f'  Weighted contributions: smooth={lambda_smooth_val.numpy()*loss_smooth:.6f}, pde={lambda_pde_val.numpy()*loss_pde:.6f}')
					print()

				# Compute relative error of neural local volatility
				K_vals = KT_train[:, 1:2]
				T_vals = KT_train[:, 2:3]
				sigma_nn_ = physics.neural_sigma(K_vals, T_vals)
				
				# For exact_sigma comparison, use S_vals (spot price) since exact_sigma is σ(t, S/S0)
				S_vals = KT_train[:, 0:1]
				t_exact = T_vals
				x_exact = S_vals / S0
				sigma_exact_ = processdata.exact_sigma(t_exact, x_exact)
				
				error_sigma = tf.reduce_mean(tf.abs(sigma_exact_ - tf.reshape(sigma_nn_, [-1, 1])) / (sigma_exact_ + 1e-6))
				rmse_sigma = tf.sqrt(tf.reduce_mean(tf.square(1 - tf.reshape(sigma_nn_, [-1, 1]) / (sigma_exact_ + 1e-6))))
				error_sigma_list.append(error_sigma)
				rmse_sigma_list.append(rmse_sigma)

				if iter_ % print_epochs == 0:
					rmse_fit = tf.sqrt(tf.reduce_mean(tf.square(physics.combined_forward(S_train, K_nn, T_nn)[0] - tf.squeeze(phi_ref))))
					weighted_smooth = lambda_smooth_val.numpy() * loss_smooth_list[-1]
					weighted_pde = lambda_pde_val.numpy() * loss_pde_list[-1]
					total_loss_val = loss_price_list[-1] + weighted_smooth + weighted_pde
					
					print(f'iter = {iter_}:')
					print(f'  Losses: price={loss_price_list[-1]:.6f}, smooth={loss_smooth_list[-1]:.6f}, pde={loss_pde_list[-1]:.6f}')
					print(f'  Weights: lambda_smooth={lambda_smooth_val.numpy():.6f}, lambda_pde={lambda_pde_val.numpy():.6f}')
					print(f'  Weighted contributions: smooth={weighted_smooth:.6f}, pde={weighted_pde:.6f}, total={total_loss_val:.6f}')
					print(f'  Metrics: error_sigma={error_sigma_list[-1]:.6f}, rmse_fit={rmse_fit:.6f}')

				if iter_ % 2000 == 0 and iter_ != 0:
					learning_rate_ /= 1.1
					physics.optimizer_NN_sigma.learning_rate.assign(learning_rate_)
					physics.optimizer_NN_price.learning_rate.assign(learning_rate_)

				if iter_ % save_epochs == 0 and iter_ != 0:
					plotter.plot_res(loss_price_list, loss_smooth_list, loss_pde_list, error_sigma_list, iter_)
					if iter_ > int(num_epochs-1):
						processdata.save_nn(physics.NN_sigma, physics.NN_price, iter_)

			return rmse_sigma_list, error_sigma_list

		def make_plots(self, rmse_sigma_list, error_sigma_list):
			print(f'relative rmse at the end of the training = {rmse_sigma_list[-1]}')
			print(f'smallest relative rmse during the training = {tf.reduce_min(rmse_sigma_list)}')

			print(f'relative error at the end of the training = {error_sigma_list[-1]}')
			print(f'smallest relative error during the training = {tf.reduce_min(error_sigma_list)}')

			# save model at the end of the training
			processdata.save_nn(physics.NN_sigma, physics.NN_price, 'final')

			fig = plt.figure(dpi=450)
			plt.semilogy(rmse_sigma_list, label='rmse')
			plt.semilogy(error_sigma_list, label='error')
			plt.legend(loc='upper right')
			plt.savefig(os.path.join(dirname, f'errors_final.png'))
			if show:
				plt.show()
			plt.close()

	trainer = Trainer()

	trainer.test()
	rmse_sigma_list, error_sigma_list = trainer.run()
	trainer.make_plots(rmse_sigma_list, error_sigma_list)

if __name__ == '__main__':
	main()

