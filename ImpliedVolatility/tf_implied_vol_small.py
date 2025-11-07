import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from scipy.integrate import ode
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import shutil
import logging, os
import datetime
import math

tf.random.set_seed(42)

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
hidden_sigma = [64, 64, 64]
hidden_price = [128, 128, 128]
lambda_smooth = 1e-3
lr = 1e-3

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

# ----------------------------
# 1️⃣ SigmaNet — implied volatility estimator
# ----------------------------
def build_sigma_net(hidden_sizes=[64, 64, 64]):
    inputs = tf.keras.Input(shape=(2,))   # K, T
    x = inputs
    for h in hidden_sizes:
        x = layers.Dense(h, activation='relu',
                         kernel_regularizer=regularizers.l2(1e-5))(x)
    sigma_out = layers.Dense(1, activation='softplus')(x)  # ensure sigma > 0
    return Model(inputs, sigma_out, name="SigmaNet")

# ----------------------------
# 2️⃣ PriceNet — price estimator using sigma + other vars
# ----------------------------
def build_price_net(input_dim, hidden_sizes=[128, 128, 128]):
    inputs = tf.keras.Input(shape=(input_dim,))  # e.g. K, T, sigma, r, S0, etc.
    x = inputs
    for h in hidden_sizes:
        x = layers.Dense(h, activation='relu',
                         kernel_regularizer=regularizers.l2(1e-5))(x)
    price_out = layers.Dense(1, activation='linear')(x)  # predicted price
    return Model(inputs, price_out, name="PriceNet")

# ----------------------------
# 3️⃣ Combined model logic
# ----------------------------
class LocalVolModel(tf.keras.Model):
    def __init__(self, hidden_sigma=[64, 64, 64], hidden_price=[128, 128, 128]):
        super().__init__()
        self.sigma_net = build_sigma_net(hidden_sigma)
        self.price_net = build_price_net(input_dim=3, hidden_sizes=hidden_price)  # [K, T, sigma]

    def call(self, inputs):
        K, T = inputs[:, 0:1], inputs[:, 1:2]
        sigma = self.sigma_net(tf.concat([K, T], axis=1))
        price_input = tf.concat([K, T, sigma], axis=1)
        price = self.price_net(price_input)
        return price, sigma

def custom_loss(y_true, y_pred, sigma, K, T, model, lambda_smooth=1e-3):
    # MSE on prices
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Smoothness regularization on sigma (∂²σ/∂K² + ∂²σ/∂T²)
    # We need to recompute sigma from K, T to get gradients
    KT_input = tf.concat([K, T], axis=1)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([K, T])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([K, T])
            sigma_pred = model.sigma_net(KT_input)
        
        # First derivatives
        dsigma_dK = tape1.gradient(sigma_pred, K)
        dsigma_dT = tape1.gradient(sigma_pred, T)
    
    # Second derivatives
    d2sigma_dK2 = tape2.gradient(dsigma_dK, K)
    d2sigma_dT2 = tape2.gradient(dsigma_dT, T)
    
    # Smoothness loss (penalize large second derivatives)
    smooth_loss = tf.reduce_mean(tf.square(d2sigma_dK2)) + tf.reduce_mean(tf.square(d2sigma_dT2))
    
    total_loss = data_loss + lambda_smooth * smooth_loss
    return total_loss

def main():
    print('Starting Implied Volatility Training')
    print('='*50)
    
    identifier = f'implied_vol_small_dataset'
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
    KT_train = tf.concat([K_nn, T_nn], axis=1)
    P_train = phi_ref
    
    print(f'Training data shape: KT={KT_train.shape}, P={P_train.shape}')
    print(f'K range: [{tf.reduce_min(K_nn):.2f}, {tf.reduce_max(K_nn):.2f}]')
    print(f'T range: [{tf.reduce_min(T_nn):.2f}, {tf.reduce_max(T_nn):.2f}]')
    print()
    
    # Initialize model
    model = LocalVolModel(hidden_sigma=hidden_sigma, hidden_price=hidden_price)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Test forward pass
    _ = model(KT_train[:1])
    print('Model initialized successfully')
    print()
    
    # Training loop
    loss_history = []
    rmse_history = []
    
    def train_step(KT, y_true):
        with tf.GradientTape() as tape:
            price_pred, sigma_pred = model(KT)
            K_vals = KT[:, 0:1]
            T_vals = KT[:, 1:2]
            loss = custom_loss(y_true, price_pred, sigma_pred, K_vals, T_vals, model, lambda_smooth)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    print('Starting training...')
    print('='*50)
    
    for epoch in range(num_epochs + 1):
        loss = train_step(KT_train, P_train)
        loss_history.append(float(loss.numpy()))
        
        # Calculate RMSE
        price_pred, _ = model(KT_train)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(P_train - price_pred)))
        rmse_history.append(float(rmse.numpy()))
        
        if epoch % print_epochs == 0:
            print(f'Epoch {epoch}: Loss = {loss.numpy():.6f}, RMSE = {rmse.numpy():.6f}')
        
        if epoch % save_epochs == 0 and epoch != 0:
            # Save model
            model.sigma_net.save(f'{dirname}/sigma_net_{epoch}.keras')
            model.price_net.save(f'{dirname}/price_net_{epoch}.keras')
            
            # Generate plots
            plot_results(model, KT_train, P_train, T_nn, K_nn, dirname, epoch, loss_history, rmse_history)
    
    # Final save
    model.sigma_net.save(f'{dirname}/sigma_net_final.keras')
    model.price_net.save(f'{dirname}/price_net_final.keras')
    plot_results(model, KT_train, P_train, T_nn, K_nn, dirname, 'final', loss_history, rmse_history)
    
    # Save loss history
    df = pd.DataFrame({'epoch': range(len(loss_history)), 'loss': loss_history, 'rmse': rmse_history})
    df.to_csv(f'{dirname}/training_history.csv', index=False)
    
    print('='*50)
    print('Training completed!')
    print(f'Final Loss: {loss_history[-1]:.6f}')
    print(f'Final RMSE: {rmse_history[-1]:.6f}')
    print(f'Results saved in: {dirname}')

def plot_results(model, KT_train, P_train, T_nn, K_nn, dirname, epoch, loss_history, rmse_history):
    """Generate visualization plots"""
    
    # Create grid for surface plots
    K_min, K_max = float(tf.reduce_min(K_nn)), float(tf.reduce_max(K_nn))
    T_min, T_max = float(tf.reduce_min(T_nn)), float(tf.reduce_max(T_nn))
    
    K_grid = np.linspace(K_min, K_max, N_)
    T_grid = np.linspace(T_min, T_max, m_)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
    
    KT_points = np.stack([K_mesh.flatten(), T_mesh.flatten()], axis=1)
    KT_tensor = tf.convert_to_tensor(KT_points, dtype=tf.float32)
    
    _, sigma_vals = model(KT_tensor)
    price_pred, _ = model(KT_tensor)
    
    sigma_surface = sigma_vals.numpy().reshape(K_mesh.shape)
    price_surface = price_pred.numpy().reshape(K_mesh.shape)
    
    # Plot 1: Implied Volatility Surface
    fig = plt.figure(figsize=(12, 5), dpi=450)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(K_mesh, T_mesh, sigma_surface, cmap=cm.inferno, linewidth=0)
    ax1.set_xlabel('Strike (K)', fontsize=12)
    ax1.set_ylabel('Maturity (T)', fontsize=12)
    ax1.set_zlabel('Implied Volatility (σ)', fontsize=12)
    ax1.set_title('Learned Implied Volatility Surface', fontsize=14)
    
    # Plot 2: Option Price Surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(K_mesh, T_mesh, price_surface, cmap=cm.RdBu_r, linewidth=0)
    ax2.set_xlabel('Strike (K)', fontsize=12)
    ax2.set_ylabel('Maturity (T)', fontsize=12)
    ax2.set_zlabel('Option Price', fontsize=12)
    ax2.set_title('Predicted Option Price Surface', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{dirname}/volatility_surface_{epoch}.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot 3: Training Loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=450)
    
    ax[0].semilogy(loss_history, label='Loss')
    ax[0].set_xlabel('Epoch', fontsize=12)
    ax[0].set_ylabel('Loss (log scale)', fontsize=12)
    ax[0].set_title('Training Loss', fontsize=14)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(rmse_history, label='RMSE', color='orange')
    ax[1].set_xlabel('Epoch', fontsize=12)
    ax[1].set_ylabel('RMSE', fontsize=12)
    ax[1].set_title('Root Mean Squared Error', fontsize=14)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dirname}/training_curves_{epoch}.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot 4: Comparison with training data
    fig = plt.figure(figsize=(10, 6), dpi=450)
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of training data
    ax.scatter(K_nn.numpy().flatten(), T_nn.numpy().flatten(), P_train.numpy().flatten(), 
               c='red', s=50, alpha=0.7, label='Training Data (3×6)')
    
    # Surface plot of predictions
    ax.plot_surface(K_mesh, T_mesh, price_surface, alpha=0.5, cmap=cm.RdBu_r, linewidth=0)
    
    ax.set_xlabel('Strike (K)', fontsize=12)
    ax.set_ylabel('Maturity (T)', fontsize=12)
    ax.set_zlabel('Option Price', fontsize=12)
    ax.set_title(f'Predicted Surface vs Training Data (Epoch {epoch})', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dirname}/prediction_comparison_{epoch}.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f'Plots saved for epoch {epoch}')

if __name__ == '__main__':
    main()

