# tf_pinn_iv_surface_full.py
# Implied volatility surface learning with SigmaNet + PriceNet
# Author: Jitender x GPT-5 | 2025-11

import os, datetime, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

# ===========================
# CONFIGURATION
# ===========================
tf.random.set_seed(42)
np.random.seed(42)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print("Device:", device)

S0 = 1000.0
r = 0.04
N = 25      # maturities
m = 20      # strikes per maturity → 500 points total
M_MC = int(1e5)
N_t = 150
dt = 0.01
gaussian_noise = 0.01

save_folder = f"tf_PINN_IVSurface_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(save_folder, exist_ok=True)

# ===========================
# Black-Scholes helpers
# ===========================
def normal_cdf(x): return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

def bs_call_price(S, K, T, r, sigma):
    sigma_safe = tf.maximum(sigma, 1e-8)
    T_safe = tf.maximum(T, 1e-8)
    d1 = (tf.math.log(S/K) + (r + 0.5*sigma_safe**2)*T_safe) / (sigma_safe*tf.sqrt(T_safe))
    d2 = d1 - sigma_safe*tf.sqrt(T_safe)
    return S * normal_cdf(d1) - K * tf.exp(-r*T_safe) * normal_cdf(d2)

def vega(S, K, T, r, sigma):
    sigma_safe = tf.maximum(sigma, 1e-8)
    T_safe = tf.maximum(T, 1e-8)
    d1 = (tf.math.log(S/K) + (r + 0.5*sigma_safe**2)*T_safe) / (sigma_safe*tf.sqrt(T_safe))
    return S * tf.sqrt(T_safe) * (1.0/tf.sqrt(2*math.pi)) * tf.exp(-0.5*d1**2)

# ===========================
# True local volatility model
# ===========================
def local_vol_true(t, S):
    x = S/S0
    y = tf.sqrt(x + 0.1) * (t + 0.1)
    return 0.3 + y * tf.exp(-y)

# ===========================
# Monte Carlo simulation
# ===========================
def run_mc_paths(M=M_MC, N_t=N_t, dt=dt):
    t_grid = np.linspace(0.0, N_t*dt, N_t).astype(np.float32)
    S = np.full((N_t, M), S0, dtype=np.float32)
    dW = np.random.normal(0,1,size=(N_t-1,M)).astype(np.float32)*np.sqrt(dt)
    for i in range(N_t-1):
        t_now = t_grid[i]
        S_now = S[i]
        sigma_now = 0.3 + np.sqrt(S_now/S0 + 0.1)*(t_now+0.1)*np.exp(-np.sqrt(S_now/S0+0.1)*(t_now+0.1))
        S[i+1] = S_now + r*S_now*dt + sigma_now*S_now*dW[i]
    return t_grid, S

# ===========================
# Data Generation
# ===========================
print("Running Monte Carlo ...")
t_grid, S_paths = run_mc_paths()
maturity_idx = np.linspace(20, N_t-1, N, dtype=int)
T_selected = t_grid[maturity_idx]
K_values = np.linspace(600, 1400, m, dtype=np.float32)

C_mc = np.zeros((N,m),dtype=np.float32)
for i,idx in enumerate(maturity_idx):
    ST = S_paths[idx]
    for j,K in enumerate(K_values):
        payoff = np.maximum(ST - K, 0)
        C_mc[i,j] = np.mean(payoff)*np.exp(-r*T_selected[i])

T_train = np.repeat(T_selected.reshape(-1,1), m, axis=1).reshape(-1,1).astype(np.float32)
K_train = np.tile(K_values.reshape(1,-1),(N,1)).reshape(-1,1).astype(np.float32)
C_train = C_mc.reshape(-1,1).astype(np.float32)
S0_train = np.ones_like(K_train)*S0
r_train = np.ones_like(K_train)*r

print(f"Generated {len(K_train)} training points.")

# ===========================
# Implied volatility inversion (Newton)
# ===========================
def implied_vol_newton(C_obs,S,K,T,r,iters=200,tol=1e-8):
    sigma = tf.fill(tf.shape(C_obs), 0.25)
    for _ in range(iters):
        C_pred = bs_call_price(S,K,T,r,sigma)
        diff = C_pred - C_obs
        v = vega(S,K,T,r,sigma)
        v_safe = tf.maximum(v, 1e-8)
        sigma_new = sigma - diff/v_safe
        sigma_new = tf.clip_by_value(sigma_new, 1e-5, 5.0)
        if tf.reduce_max(tf.abs(sigma_new - sigma)) < tol:
            sigma = sigma_new; break
        sigma = sigma_new
    return sigma

print("Inverting implied vols from MC prices ...")
C_tf = tf.constant(C_train)
K_tf = tf.constant(K_train)
T_tf = tf.constant(T_train)
S_tf = tf.constant(S0_train)
r_tf = tf.constant(r_train)
sigma_gt = implied_vol_newton(C_tf,S_tf,K_tf,T_tf,r_tf,200)
sigma_gt_np = sigma_gt.numpy().reshape(N,m)

# ===========================
# Build Networks
# ===========================
def dense_block(x, units=64, act='tanh', depth=3):
    for _ in range(depth):
        x = tf.keras.layers.Dense(units, activation=act)(x)
    return x

def build_sigma_net():
    inp = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.GaussianNoise(gaussian_noise)(inp)
    x = dense_block(x,64,'tanh',3)
    out = tf.keras.layers.Dense(1,activation='softplus')(x)
    return tf.keras.Model(inp,out,name='SigmaNet')

def build_price_net():
    inp = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.GaussianNoise(gaussian_noise)(inp)
    x = dense_block(x,64,'tanh',3)
    out = tf.keras.layers.Dense(1,activation='softplus')(x)
    return tf.keras.Model(inp,out,name='PriceNet')

sigma_net = build_sigma_net()
price_net = build_price_net()

# ===========================
# Loss components
# ===========================
def bs_pde_residual(S0,K,T,r):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([S0,K,T])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([S0,K,T])
            sigma = sigma_net(tf.concat([K,T],1))
            C = price_net(tf.concat([S0,K,T,sigma,r],1))
        dC_dT = t1.gradient(C,T)
        dC_dS = t1.gradient(C,S0)
    d2C_dS2 = t2.gradient(dC_dS,S0)
    del t1,t2
    res = dC_dT + 0.5*tf.square(sigma)*tf.square(S0)*d2C_dS2 + r*S0*dC_dS - r*C
    return res

def arbitrage_loss(S0,K,T,r):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(K)
        with tf.GradientTape() as t1:
            t1.watch(K)
            sigma = sigma_net(tf.concat([K,T],1))
            C = price_net(tf.concat([S0,K,T,sigma,r],1))
        dC_dK = t1.gradient(C,K)
    d2C_dK2 = t2.gradient(dC_dK,K)
    mono = tf.reduce_mean(tf.square(tf.nn.relu(dC_dK)))
    conv = tf.reduce_mean(tf.square(tf.nn.relu(-d2C_dK2)))
    return mono+conv

def smoothness_loss(K,T):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([K,T])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([K,T])
            sigma = sigma_net(tf.concat([K,T],1))
        ds_dK = t1.gradient(sigma,K)
        ds_dT = t1.gradient(sigma,T)
    d2s_dK2 = t2.gradient(ds_dK,K)
    d2s_dT2 = t2.gradient(ds_dT,T)
    return tf.reduce_mean(tf.square(d2s_dK2)) + tf.reduce_mean(tf.square(d2s_dT2))

# ===========================
# Training setup
# ===========================
optimizer = tf.keras.optimizers.Adam(1e-3)
num_epochs = 10000
print_every = 500
save_every = 2500

λ_mse = tf.Variable(1.0)
λ_pde = tf.Variable(0.1)
λ_arb = tf.Variable(0.1)
λ_smooth = tf.Variable(0.01)

K_min, K_max = np.min(K_train), np.max(K_train)
T_min, T_max = np.min(T_train), np.max(T_train)

# ===========================
# Training loop
# ===========================
def compute_total_loss():
    # MSE
    sigma_pred = sigma_net(tf.concat([K_tf,T_tf],1))
    C_pred = price_net(tf.concat([S_tf,K_tf,T_tf,sigma_pred,r_tf],1))
    L_mse = tf.reduce_mean(tf.square(C_pred - C_tf))

    # PDE
    n_colloc = 512
    K_c = tf.random.uniform([n_colloc,1],K_min,K_max)
    T_c = tf.random.uniform([n_colloc,1],T_min,T_max)
    S_c = tf.ones_like(K_c)*S0
    r_c = tf.ones_like(K_c)*r
    res = bs_pde_residual(S_c,K_c,T_c,r_c)
    L_pde = tf.reduce_mean(tf.square(res))

    # Arbitrage + smoothness
    L_arb = arbitrage_loss(S_c,K_c,T_c,r_c)
    L_smooth = smoothness_loss(K_c,T_c)

    L_total = λ_mse*L_mse + λ_pde*L_pde + λ_arb*L_arb + λ_smooth*L_smooth
    return L_total,L_mse,L_pde,L_arb,L_smooth

hist = {'total':[],'mse':[],'pde':[],'arb':[],'smooth':[]}
start=time.time()
for ep in range(1,num_epochs+1):
    with tf.GradientTape() as tape:
        L_total,L_mse,L_pde,L_arb,L_smooth = compute_total_loss()
    grads = tape.gradient(L_total, sigma_net.trainable_variables + price_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, sigma_net.trainable_variables + price_net.trainable_variables))
    hist['total'].append(float(L_total)); hist['mse'].append(float(L_mse))
    hist['pde'].append(float(L_pde)); hist['arb'].append(float(L_arb)); hist['smooth'].append(float(L_smooth))
    if ep%print_every==0:
        print(f"Ep {ep:5d}: L={L_total:.3e} | MSE={L_mse:.3e} | PDE={L_pde:.3e}")

end=time.time()
print("Training time (min):", (end-start)/60)

# ===========================
# Evaluation & plots
# ===========================
sigma_pred = sigma_net(tf.concat([K_tf,T_tf],1)).numpy().reshape(N,m)
C_pred = price_net(tf.concat([S_tf,K_tf,T_tf,sigma_gt,r_tf],1)).numpy().reshape(N,m)

price_rmse = np.sqrt(np.mean((C_pred - C_mc)**2))
vol_rmse = np.sqrt(np.mean((sigma_pred - sigma_gt_np)**2))

print(f"Final Price RMSE={price_rmse:.4e}, Vol RMSE={vol_rmse:.4e}")

pd.DataFrame({'Metric':['Price_RMSE','Vol_RMSE'],'Value':[price_rmse,vol_rmse]}).to_csv(os.path.join(save_folder,'metrics.csv'),index=False)

fig = plt.figure(figsize=(16,6))
K_mesh,T_mesh=np.meshgrid(K_values,T_selected)
ax=fig.add_subplot(131,projection='3d');ax.plot_surface(K_mesh,T_mesh,sigma_pred,cmap='plasma');ax.set_title('Learned σ(K,T)')
ax=fig.add_subplot(132,projection='3d');ax.plot_surface(K_mesh,T_mesh,sigma_gt_np,cmap='plasma');ax.set_title('Ground Truth IV (Newton)')
ax=fig.add_subplot(133,projection='3d');ax.plot_surface(K_mesh,T_mesh,sigma_pred-sigma_gt_np,cmap='RdBu_r');ax.set_title('Error')
plt.savefig(os.path.join(save_folder,'vol_surface_comparison.png'));plt.close()

plt.figure();plt.semilogy(hist['total']);plt.title('Total Loss');plt.savefig(os.path.join(save_folder,'loss_curve.png'));plt.close()

sigma_net.save(os.path.join(save_folder,'sigma_net.keras'))
price_net.save(os.path.join(save_folder,'price_net.keras'))
print("Saved to:", save_folder)
