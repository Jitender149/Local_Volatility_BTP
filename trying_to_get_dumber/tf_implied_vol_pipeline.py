# tf_implied_vol_pipeline.py
# Run with: python tf_implied_vol_pipeline.py
import os, datetime, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

# -----------------------
# Config
# -----------------------
tf.random.set_seed(42)
np.random.seed(42)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print("Device:", device)

S0 = 1000.0
r = 0.04

N = 10   # maturities
m = 20   # strikes per maturity (=> 200 training points)
M_MC = int(2e5)   # MC paths (reduce for speed: 5e4, 1e5)
N_t = 150
dt = 0.01

save_folder = f"tf_iv_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(save_folder, exist_ok=True)

# -----------------------
# Helper: black scholes (TensorFlow)
# -----------------------
def normal_cdf_tf(x):
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

def bs_call_price_tf(S, K, T, r, sigma):
    # shapes: (batch,1) or scalars
    sigma_safe = tf.maximum(sigma, 1e-8)
    T_safe = tf.maximum(T, 1e-8)
    d1 = (tf.math.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * tf.sqrt(T_safe))
    d2 = d1 - sigma_safe * tf.sqrt(T_safe)
    return S * normal_cdf_tf(d1) - K * tf.exp(-r * T_safe) * normal_cdf_tf(d2)

def vega_tf(S, K, T, r, sigma):
    sigma_safe = tf.maximum(sigma, 1e-8)
    T_safe = tf.maximum(T, 1e-8)
    d1 = (tf.math.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * tf.sqrt(T_safe))
    return S * tf.sqrt(T_safe) * (1.0 / tf.sqrt(2*math.pi)) * tf.exp(-0.5 * d1**2)

# -----------------------
# True local volatility (used to simulate data)
# -----------------------
def local_vol_true_tf(t, S):
    # S and t are tensors/scalars; implement same shape rules
    x = S / S0
    y = tf.sqrt(x + 0.1) * (t + 0.1)
    return 0.3 + y * tf.exp(-y)   # >0

# -----------------------
# Monte Carlo simulation (Euler-Maruyama)
# -----------------------
def run_mc_paths(M=M_MC, N_t=N_t, dt=dt):
    # Generates S_t for all times (N_t) for each path (M)
    t_grid = np.linspace(0.0, N_t * dt, N_t).astype(np.float32)
    S = np.full((N_t, M), S0, dtype=np.float32)
    # Generate increments shape (N_t-1, M)
    dW = np.random.normal(0.0, 1.0, size=(N_t-1, M)).astype(np.float32) * np.sqrt(dt)
    for i in range(N_t-1):
        t_now = t_grid[i]
        S_now = S[i]
        sigma_now = (np.sqrt(S_now / S0 + 0.1) * (t_now + 0.1) * np.exp(-np.sqrt(S_now / S0 + 0.1) * (t_now + 0.1))) + 0.3
        # Euler-Maruyama
        S[i+1] = S_now + r * S_now * dt + sigma_now * S_now * dW[i]
    return t_grid, S

# -----------------------
# Create training grid (K,T) and compute MC prices
# -----------------------
print("Running MC (this may take time)...")
t_grid, S_paths = run_mc_paths()
maturity_indices = np.linspace(20, N_t-1, N, dtype=int)
T_selected = t_grid[maturity_indices]  # N maturities
K_values = np.linspace(600.0, 1400.0, m, dtype=np.float32)

# compute MC prices
C_mc = np.zeros((N, m), dtype=np.float32)
for i, idx in enumerate(maturity_indices):
    ST = S_paths[idx]   # length M
    for j, K in enumerate(K_values):
        payoff = np.maximum(ST - K, 0.0)
        C_mc[i,j] = np.mean(payoff) * np.exp(-r * T_selected[i])

# Flatten training arrays
T_train = np.repeat(T_selected.reshape(-1,1), m, axis=1).reshape(-1,1).astype(np.float32)
K_train = np.tile(K_values.reshape(1,-1), (N,1)).reshape(-1,1).astype(np.float32)
C_train = C_mc.reshape(-1,1).astype(np.float32)
S0_train = np.ones_like(T_train) * S0
r_train = np.ones_like(T_train) * r

print("Training points:", K_train.shape[0])  # should be N*m

# -----------------------
# Newton inversion: implied vol from MC prices (vectorized TF)
# -----------------------
def implied_vol_newton_tf(C_obs, S, K, T, r, iters=50, tol=1e-8):
    # C_obs, S, K, T shape: (n,1)
    # initial guess: ATM-ish
    sigma = tf.fill(tf.shape(C_obs), 0.25)
    for i in range(iters):
        price = bs_call_price_tf(S, K, T, r, sigma)
        diff = price - C_obs
        v = vega_tf(S, K, T, r, sigma)
        # avoid tiny vega
        v_safe = tf.maximum(v, 1e-8)
        sigma_new = sigma - diff / v_safe
        # clip to positive
        sigma_new = tf.clip_by_value(sigma_new, 1e-6, 5.0)
        if tf.reduce_max(tf.abs(sigma_new - sigma)) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new
    return sigma

print("Computing implied vol from MC prices...")
C_obs_tf = tf.constant(C_train, dtype=tf.float32)
S_tf = tf.constant(S0_train, dtype=tf.float32)
K_tf = tf.constant(K_train, dtype=tf.float32)
T_tf = tf.constant(T_train, dtype=tf.float32)
r_tf = tf.constant(r_train, dtype=tf.float32)
sigma_gt = implied_vol_newton_tf(C_obs_tf, S_tf, K_tf, T_tf, r_tf, iters=200)
sigma_gt_np = sigma_gt.numpy().reshape(N,m)

# Save ground-truth implied vol surface
df = pd.DataFrame({
    'K': K_train.flatten(),
    'T': T_train.flatten(),
    'C_mc': C_train.flatten(),
    'sigma_gt': sigma_gt.numpy().flatten()
})
df.to_csv(os.path.join(save_folder, 'mc_prices_and_iv_ground_truth.csv'), index=False)

# quick plot of GT implied vol
fig = plt.figure(figsize=(8,5))
plt.scatter(K_train.flatten(), sigma_gt.numpy().flatten(), c=T_train.flatten(), cmap='viridis', s=25)
plt.colorbar(label='T')
plt.xlabel('K'); plt.ylabel('Implied sigma (GT)')
plt.title('Ground-truth implied vol (from MC prices)')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'gt_iv_scatter.png')); plt.close()

# -----------------------
# Build SigmaNet (K,T) -> sigma
# -----------------------
from tensorflow.keras import layers, models, optimizers

def build_sigma_net(num_neurons=64, num_layers=4, activation='tanh'):
    inp = layers.Input(shape=(2,))
    x = inp
    for _ in range(num_layers):
        x = layers.Dense(num_neurons, activation=activation)(x)
    out = layers.Dense(1, activation='softplus')(x)  # positive vol
    model = models.Model(inp, out, name='SigmaNet')
    return model

sigma_net = build_sigma_net()
sigma_net.summary()

# optimizer and training params
optimizer = optimizers.Adam(1e-3)
batch_size = 64
epochs = 4000  # adjust as needed
print_every = 200

# Dataset
KT = np.hstack([K_train, T_train]).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((KT, C_train))
dataset = dataset.shuffle(200).batch(batch_size).prefetch(2)

# Loss: BS price MSE
@tf.function
def train_step(KT_batch, C_obs_batch):
    with tf.GradientTape() as tape:
        Kb = tf.reshape(KT_batch[:,0:1], (-1,1))
        Tb = tf.reshape(KT_batch[:,1:2], (-1,1))
        sigma_pred = sigma_net(tf.concat([Kb, Tb], axis=1))
        C_pred = bs_call_price_tf(tf.ones_like(Kb)*S0, Kb, Tb, tf.ones_like(Kb)*r, sigma_pred)
        loss = tf.reduce_mean(tf.square(C_pred - C_obs_batch))
        # smoothness regularizer (optional)
        # compute grads of sigma_pred w.r.t inputs for small batch if you want
    grads = tape.gradient(loss, sigma_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, sigma_net.trainable_variables))
    return loss

# train
loss_hist = []
start = time.time()
for ep in range(1, epochs+1):
    epoch_loss = 0.0
    nb = 0
    for KT_b, C_b in dataset:
        l = train_step(KT_b, C_b)
        epoch_loss += l.numpy()
        nb += 1
    avg_loss = epoch_loss / nb
    loss_hist.append(avg_loss)
    if ep % print_every == 0 or ep == 1:
        # evaluate RMSE price and vol
        sigma_pred_all = sigma_net(KT).numpy().flatten()
        C_pred_all = bs_call_price_tf(tf.ones_like(K_tf)*S0, K_tf, T_tf, tf.ones_like(K_tf)*r, tf.constant(sigma_pred_all.reshape(-1,1), dtype=tf.float32)).numpy().flatten()
        rmse_price = np.sqrt(np.mean((C_pred_all - C_train.flatten())**2))
        # vol rmse versus GT (note GT computed by Newton)
        vol_rmse = np.sqrt(np.mean((sigma_pred_all - sigma_gt.numpy().flatten())**2))
        print(f"Ep {ep:4d} loss={avg_loss:.6e} price_RMSE={rmse_price:.6e} vol_RMSE_vsGT={vol_rmse:.6e}")

end = time.time()
print("Training time (s):", end-start)

# final predictions
sigma_pred_final = sigma_net(KT).numpy().reshape(N,m)
C_pred_final = bs_call_price_tf(tf.ones_like(K_tf)*S0, K_tf, T_tf, tf.ones_like(K_tf)*r, tf.constant(sigma_pred_final.reshape(-1,1), dtype=tf.float32)).numpy().reshape(N,m)

# Save results & metrics
np.save(os.path.join(save_folder,'sigma_gt.npy'), sigma_gt_np)
np.save(os.path.join(save_folder,'sigma_pred.npy'), sigma_pred_final)
np.save(os.path.join(save_folder,'C_mc.npy'), C_mc)
np.save(os.path.join(save_folder,'C_pred.npy'), C_pred_final)

metrics = {
    'price_rmse': float(np.sqrt(np.mean((C_pred_final - C_mc)**2))),
    'vol_rmse': float(np.sqrt(np.mean((sigma_pred_final - sigma_gt_np)**2)))
}
pd.DataFrame([metrics]).to_csv(os.path.join(save_folder,'final_metrics.csv'), index=False)

# plots: surface comparison
K_plot = K_values
T_plot = T_selected
K_mesh, T_mesh = np.meshgrid(K_plot, T_plot)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(18,5))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(K_mesh, T_mesh, sigma_pred_final, cmap='plasma'); ax.set_title('Learned implied sigma')
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(K_mesh, T_mesh, sigma_gt_np, cmap='plasma'); ax.set_title('GT implied sigma (from Newton)')
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(K_mesh, T_mesh, sigma_pred_final - sigma_gt_np, cmap='RdBu_r'); ax.set_title('Difference')
plt.savefig(os.path.join(save_folder, 'sigma_surface_comparison.png')); plt.close()

fig2 = plt.figure(figsize=(12,4))
ax1 = fig2.add_subplot(121)
ax1.semilogy(loss_hist)
ax1.set_title('Training loss (price-MSE)')
ax2 = fig2.add_subplot(122)
ax2.plot(sigma_gt_np.flatten(), sigma_pred_final.flatten(), '.')
ax2.plot([0,1],[0,1],'r--'); ax2.set_title('Pred vs GT implied')
plt.savefig(os.path.join(save_folder, 'loss_and_scatter.png')); plt.close()

# Save model
sigma_net.save(os.path.join(save_folder, 'sigma_net_tf.keras'))

print("Saved results in:", save_folder)
