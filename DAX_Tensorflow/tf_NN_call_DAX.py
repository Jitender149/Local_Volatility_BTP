import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
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

VariableSpec = resource_variable_ops.VariableSpec

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if len(gpus)!=0:
    device = gpus[0]
    tf.config.set_visible_devices(device, 'GPU')
    tf.config.experimental.set_memory_growth(device, True)
else:
    device = cpus[0]
    tf.config.set_visible_devices(device, 'CPU')

print('Simulation is running on:')
print(f'device = {device}')
print()

data_type    = tf.float32
data_type_nn = tf.float32
tf.keras.backend.set_floatx('float32')

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

show = False
num_epochs = 30000
print_epochs = 2500
save_epochs = 10000

dates_S0 = [['7_August_2001',5752.51], ['8_August_2001',5614.51], ['9_August_2001',5512.28]]
ldups = [1,0]

#NN Params
gaussian_phi = 0.5
gaussian_eta = 0.5
num_res_blocks = 3
div_lr = 10
lr = 10**-3

#Plotting params
N = 256
m = 256

#MC Params
#risk free rate
r_ = 0.04
# number of samples
M  = 10**4
# number of time steps
N_t = 1000
# size of time step
dt = 10**-3

all_repricings = []

def main(ldup, date_S0):            

    repricings = []
    print (date_S0, ldup)
    print ('')
    
    date_ = date_S0[0]
    S0 = date_S0[1] 

    data_filename = f'dataTrain_{date_}.csv'
    data_path = os.path.join('.', data_filename)
    identifier = f'batch_call_DAX_{date_.split("_")[0]}aug_ldup_{ldup}'

    folder_name_load = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    dirname = folder_name_save

    S_0 = tf.constant(S0, dtype=data_type)
    r = tf.constant(r_,    dtype=data_type)

    class LoadData:
        def __init__(self):
            super(LoadData, self).__init__()
            self.option_type = 1

        def save_nn(self, NN_phi, NN_eta, iter_=None):

            tf.keras.models.save_model(NN_phi,
                                       filepath = f'{folder_name_save}/NN_phi_{iter_}.keras',
                                       overwrite = True)

            tf.keras.models.save_model(NN_eta,
                                       filepath = f'{folder_name_save}/NN_eta_{iter_}.keras',
                                       overwrite = True)

        def load_nn(self, folder_name_load, iter_=None):

            NN_phi = tf.keras.models.load_model(f'{folder_name_load}/NN_phi_{iter_}.keras')

            NN_eta = tf.keras.models.load_model(f'{folder_name_load}/NN_eta_{iter_}.keras')

            return NN_phi, NN_eta

        def save_phi_exact(self, phi_exact):
            phi_exact_ = tf.data.Dataset.from_tensor_slices(phi_exact)
            tf.data.experimental.save(phi_exact_, f'{folder_name_save}/phi_exact')

        def load_phi_exact(self, folder_name_load):
            tensorspec = (tf.TensorSpec(shape=None, dtype=data_type, name=None),)
            phi_loaded   = tf.data.experimental.load(f'{folder_name_load}/phi_exact', tensorspec)

            phi_exact_list = []
            for item in phi_loaded:
                phi_exact_list.append(item)

            phi_exact = tf.concat(phi_exact_list, axis=0)

            return np.array(phi_exact)

        def read_csv(self, data_filename):

            phi_df = pd.read_csv(data_filename)

            ids,_ = np.where(tf.reshape(tf.cast(phi_df['Option\ntype'], dtype=data_type), [-1,1]) == self.option_type)
            T_nn = tf.gather(tf.reshape(tf.cast(phi_df['Maturity'], dtype=data_type), [-1,1]), ids)
            K_nn = tf.gather(tf.reshape(tf.cast(phi_df['Strike'], dtype=data_type), [-1,1]), ids)
            phi_ref = tf.gather(tf.reshape(tf.cast(phi_df['Option\nprice'], dtype=data_type), [-1,1]), ids)
            sigma_loc_ref = tf.gather(tf.reshape(tf.cast(phi_df['Implied\nvol.'], dtype=data_type), [-1,1]), ids)

            return T_nn, K_nn, phi_ref, sigma_loc_ref, ids

        def get_min_max(self, T_nn, K_nn):
            t_min = tf.reduce_min(T_nn)
            t_max = tf.reduce_max(T_nn)
            k_min = tf.reduce_min(K_nn)
            k_max = tf.reduce_max(K_nn)

            self.t_max = t_max
            self.k_max = k_max

            return t_min, t_max, k_min, k_max

        def scale_data(self, T_nn, K_nn):

            t_nn = T_nn / self.t_max
            k_nn = tf.exp(-r*T_nn) * K_nn / self.k_max

            if show:
                print(f't_min = {tf.reduce_min(t_nn)}, t_max = {tf.reduce_max(t_nn)}, k_min = {tf.reduce_min(k_nn)}, k_max = {tf.reduce_max(k_nn)}')
                
            t_tilde = t_nn
            k_tilde = k_nn

            x = [t_tilde, k_tilde]

            return x

        def get_min_max_random(self, t_nn, k_nn, T_nn):
            t_min_random = tf.reduce_min(t_nn).numpy()
            t_max_random = tf.reduce_max(t_nn).numpy()
            k_min_random = tf.reduce_min(k_nn).numpy()
            k_max_random = tf.reduce_max(tf.exp(-r*T_nn)).numpy()

            print (f't_min_random = {t_min_random}, t_max_random = {t_max_random}')
            print (f'k_min_random = {k_min_random}, k_max_random = {k_max_random}')

            t_min_random = math.floor(tf.reduce_min(t_nn).numpy()*100)/100
            t_max_random = math.ceil(tf.reduce_max(t_nn).numpy()*100)/100
            k_min_random = math.floor(tf.reduce_min(k_nn).numpy()*100)/100
            k_max_random = math.ceil(tf.reduce_max(tf.exp(-r*T_nn)).numpy()*100)/100

            print (f't_min_random = {t_min_random}, t_max_random = {t_max_random}')
            print (f'k_min_random = {k_min_random}, k_max_random = {k_max_random}')

            x = [t_min_random, t_max_random, k_min_random, k_max_random]

            return x

    processdata = LoadData()

    T_nn, K_nn, phi_ref, sigma_loc_ref, ids = processdata.read_csv(data_filename)
    processdata.save_phi_exact(phi_ref)
    phi_tilde_ref = phi_ref / S_0

    t_min, t_max, k_min, k_max = processdata.get_min_max(T_nn, K_nn)
    x = processdata.scale_data(T_nn, K_nn)
    t_tilde, k_tilde = x 
    x_random = processdata.get_min_max_random(t_tilde, k_tilde, T_nn)
    [t_min_random, t_max_random, k_min_random, k_max_random] = x_random

    class PhysicsModel(tf.keras.Model):
        def __init__(self, lambda_pde=1.0):
            super(PhysicsModel, self).__init__()
            self.lambda_pde = lambda_pde
            self.lambda_reg = 1.0
            self.num_res_blocks = num_res_blocks
            self.activation = 'tanh'
            self.gaussian_phi = gaussian_phi
            self.gaussian_eta = gaussian_eta

        def residual_block(self, input_tensor, units=64, activation='tanh'):
            """
            Defines a single residual block with two Dense-BatchNorm-Activation layers 
            and a residual (skip) connection.
            """
            dense_1 = tf.keras.layers.Dense(units, use_bias=False)(input_tensor)
            batchnorm_1 = tf.keras.layers.BatchNormalization()(dense_1)
            activation_1 = tf.keras.layers.Activation(activation)(batchnorm_1)

            dense_2 = tf.keras.layers.Dense(units, use_bias=False)(activation_1)
            batchnorm_2 = tf.keras.layers.BatchNormalization()(dense_2)
            activation_2 = tf.keras.layers.Activation(activation)(batchnorm_2)

            # Residual connection (skip connection)
            added = tf.keras.layers.Add()([input_tensor, activation_2])
            return added

        def net_phi_tilde(self, num_res_blocks=5, units=64, activation='tanh'):
            """
            Builds a neural network model with a customizable number of residual blocks, 
            mapping (T, K) to an option price phi in R+ with softplus activation.

            Args:
                num_res_blocks (int): Number of residual blocks to include in the model.
                units (int): Number of units in each Dense layer of the residual block.
            """
            input_ = tf.keras.Input(shape=(2,))

            noisy_input = tf.keras.layers.GaussianNoise(self.gaussian_phi)(input_)
            dense_in = tf.keras.layers.Dense(units, activation=activation, use_bias=False)(noisy_input)

            # Apply the specified number of residual blocks
            x = dense_in
            for _ in range(num_res_blocks):
                x = self.residual_block(x, units, activation)

            # Final dense layers
            dense_out = tf.keras.layers.Dense(units, activation=activation, use_bias=True)(x)
            output_ = tf.keras.layers.Dense(1, activation='softplus', use_bias=True, dtype="float32")(dense_out)

            model = tf.keras.models.Model(inputs=input_, outputs=output_)
            return model

        def net_eta_tilde(self, num_res_blocks=5, units=64, activation='tanh'):
            """
            Builds a neural network model with a customizable number of residual blocks, 
            mapping (T, K) to an option price phi in R+ with softplus activation.

            Args:
                num_res_blocks (int): Number of residual blocks to include in the model.
                units (int): Number of units in each Dense layer of the residual block.
            """
            input_ = tf.keras.Input(shape=(2,))

            noisy_input = tf.keras.layers.GaussianNoise(self.gaussian_eta)(input_)
            dense_in = tf.keras.layers.Dense(units, activation='tanh', use_bias=False)(noisy_input)

            # Apply the specified number of residual blocks
            x = dense_in
            for _ in range(num_res_blocks):
                x = self.residual_block(x, units, activation)

            # Final dense layers
            dense_out = tf.keras.layers.Dense(units, activation=activation, use_bias=True)(x)
            output_ = tf.keras.layers.Dense(1, activation='softplus', use_bias=True, dtype="float32")(dense_out)

            model = tf.keras.models.Model(inputs=input_, outputs=output_)
            return model

        def build_models(self):

            self.NN_phi_tilde = self.net_phi_tilde(num_res_blocks=self.num_res_blocks, units=64, activation=self.activation)
            self.NN_eta_tilde = self.net_eta_tilde(num_res_blocks=self.num_res_blocks, units=64, activation=self.activation)

            self.optimizer_NN_phi = tf.keras.optimizers.Adam(learning_rate = 10**-4)
            self.optimizer_NN_eta = tf.keras.optimizers.Adam(learning_rate = 10**-4)

        def neural_phi_tilde(self, t_tilde, k_tilde):
            phi_nn_ = self.NN_phi_tilde(tf.concat([t_tilde, k_tilde], axis=1))
            # return (1 - tf.exp((k_tilde - 1) * phi_nn_))
            return (1 - tf.exp(-phi_nn_))

        def neural_eta_tilde(self, t_tilde, k_tilde):
            eta_nn_ = self.NN_eta_tilde(tf.concat([t_tilde, k_tilde], axis=1))
            return eta_nn_

        def neural_phi(self, T, K):
            T_nn = tf.cast(tf.reshape(T, [-1,1]), dtype=data_type)
            K_nn = tf.cast(tf.reshape(K, [-1,1]), dtype=data_type)
            x = processdata.scale_data(T_nn, K_nn)
            t_tilde, k_tilde = x 
            phi_nn_ = S_0 * self.neural_phi_tilde(t_tilde, k_tilde)
            return phi_nn_

        def neural_sigma(self, T, K):
            T_nn = tf.cast(tf.reshape(T, [-1,1]), dtype=data_type)
            K_nn = tf.cast(tf.reshape(K, [-1,1]), dtype=data_type)
            x = processdata.scale_data(T_nn, K_nn)
            t_tilde, k_tilde = x 
            sigma_nn = tf.sqrt(2*self.neural_eta_tilde(t_tilde, k_tilde)/(t_max))
            return tf.squeeze(sigma_nn)

        def clip(self, y):
            x = tf.stop_gradient(tf.reduce_mean(y**2)/y**2)
            return tf.clip_by_value(x, clip_value_min=0.1, clip_value_max=10)

        def weight(self, y):
            return 1 + self.clip(y)/tf.reduce_mean(self.clip(y))

        def loss_phi_cal(self):
            phi_tilde_nn = self.neural_phi_tilde(t_tilde, k_tilde)
            loss_phi_ = tf.reduce_mean(self.weight(phi_tilde_ref) * tf.square(phi_tilde_nn - phi_tilde_ref))

            # impose boundary condition
            t_tilde_0 = tf.cast(tf.reshape(np.full(128, 0), [-1,1]), dtype=data_type)
            k_tilde_0 = tf.random.uniform(shape = [128, 1], minval=k_min_random, maxval=k_max_random, dtype=data_type)
            phi_tilde_0 = tf.nn.relu(1 - (1/S_0) * k_max*k_tilde_0)
            loss_bc_ = tf.reduce_mean(self.weight(phi_tilde_0) * tf.square((self.neural_phi_tilde(t_tilde_0, k_tilde_0) - phi_tilde_0)))

            return loss_phi_ + loss_bc_

        def loss_dupire_cal(self):

            t_tilde_0 = tf.cast(tf.reshape(np.full(128, t_min_random), [-1,1]), dtype=data_type)
            t_tilde_1 = tf.cast(tf.reshape(np.full(128, t_max_random), [-1,1]), dtype=data_type)
            k_tilde_0 = tf.cast(tf.reshape(np.full(128, k_min_random), [-1,1]), dtype=data_type)
            k_tilde_1 = tf.cast(tf.reshape(np.full(128, k_max_random), [-1,1]), dtype=data_type)
            t_tilde_bulk = tf.random.uniform(shape = [128*128, 1], minval=t_min_random, maxval=t_max_random, dtype=data_type)
            k_tilde_bulk = tf.random.uniform(shape = [128*128, 1], minval=k_min_random, maxval=k_max_random, dtype=data_type)

            t_tilde_random = tf.concat([t_tilde_0, t_tilde_1, t_tilde_bulk], axis=0)
            k_tilde_random = tf.concat([k_tilde_bulk, k_tilde_0, k_tilde_1], axis=0)

            with tf.GradientTape(persistent=True) as tape_2:
                tape_2.watch(k_tilde_random)
                with tf.GradientTape(persistent=True) as tape_1:
                    tape_1.watch(t_tilde_random)
                    tape_1.watch(k_tilde_random)

                    phi_tilde_ = self.neural_phi_tilde(t_tilde_random, k_tilde_random)

                grad_phi_t_tilde = tape_1.gradient(phi_tilde_, t_tilde_random)
                grad_phi_k_tilde = tape_1.gradient(phi_tilde_, k_tilde_random)

            grad_phi_kk_tilde = tape_2.gradient(grad_phi_k_tilde, k_tilde_random)

            eta_tilde_ = self.neural_eta_tilde(t_tilde_random, k_tilde_random)

            dupire_eqn = grad_phi_t_tilde - eta_tilde_ * k_tilde_random**2 * grad_phi_kk_tilde

            loss_dupire_ = tf.reduce_mean(self.weight(grad_phi_t_tilde) * tf.square(dupire_eqn))

            arb_eqn = grad_phi_t_tilde - r * t_max * k_tilde_random * tf.nn.relu(grad_phi_k_tilde)

            loss_reg_ = tf.reduce_mean(self.weight(grad_phi_t_tilde) * tf.square(tf.nn.relu(-arb_eqn)))

            return loss_dupire_, loss_reg_

        @tf.function
        def train_step(self, lambda_pde=None, lambda_reg=None):
            if lambda_pde == None:
                lambda_pde = self.lambda_pde
            if lambda_reg == None:
                lambda_reg = self.lambda_reg
            with tf.GradientTape(persistent=True) as tape:
                loss_phi = self.loss_phi_cal()
                loss_dupire, loss_reg = self.loss_dupire_cal()
                loss_total = loss_phi + lambda_pde * loss_dupire + lambda_reg * loss_reg

            grads_NN_phi = tape.gradient(loss_total, self.NN_phi_tilde.trainable_variables)
            grads_NN_eta = tape.gradient(loss_dupire, self.NN_eta_tilde.trainable_variables)

            self.optimizer_NN_phi.apply_gradients(zip(grads_NN_phi, self.NN_phi_tilde.trainable_variables))
            self.optimizer_NN_eta.apply_gradients(zip(grads_NN_eta, self.NN_eta_tilde.trainable_variables))

            return loss_phi, loss_dupire, loss_reg

    physics = PhysicsModel(ldup)

    physics.build_models()

    class Plotter:
        def __init__(self):
            super(Plotter, self).__init__()

        def plot_res(self, loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step):
            # Primary method to handle all plotting, taking in losses and error lists, and the current training step

            # Generate grid for plotting over a range of maturity (T) and strike prices (K)
            T_min, T_max = tf.reduce_min(T_nn), tf.reduce_max(T_nn)
            K_min, K_max = tf.reduce_min(K_nn), tf.reduce_max(K_nn)

            # Create a mesh grid of T and K for surface plots
            T_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(T_min, T_max, N), dtype=data_type), [-1, 1]), m, axis=1), [-1, 1])
            K_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(K_min, K_max, m), dtype=data_type), [1, -1]), N, axis=0), [-1, 1])

            # Reshape tensors for easier manipulation
            T_nn_ = tf.reshape(T_, [-1, 1])
            K_nn_ = tf.reshape(K_, [-1, 1])

            x_ = processdata.scale_data(T_nn_, K_nn_)
            t_tilde_, k_tilde_ = x_ 

            # Obtain model predictions for option prices and local volatility
            phi_tilde_ = physics.neural_phi_tilde(t_tilde_, k_tilde_)
            sigma_nn_ = tf.sqrt(2 * physics.neural_eta_tilde(t_tilde_, k_tilde_) / t_max)

            # Calculate gradients for the Dupire equation
            with tf.GradientTape(persistent=True) as tape_2:
                tape_2.watch(k_tilde_)
                with tf.GradientTape(persistent=True) as tape_1:
                    tape_1.watch(t_tilde_)
                    tape_1.watch(k_tilde_)
                    phi_tilde_ = physics.neural_phi_tilde(t_tilde_, k_tilde_)

                grad_phi_t_tilde = tape_1.gradient(phi_tilde_, t_tilde_)
                grad_phi_k_tilde = tape_1.gradient(phi_tilde_, k_tilde_)

            grad_phi_kk_tilde = tape_2.gradient(grad_phi_k_tilde, k_tilde_)

            # Calculate the Dupire equation error term
            eta_tilde_ = physics.neural_eta_tilde(t_tilde_, k_tilde_)
            sec_term = eta_tilde_ * (k_min / (k_max - k_min) + k_tilde_)**2 * grad_phi_kk_tilde
            dupire_eqn_error = grad_phi_t_tilde - sec_term

            ########################################################################################################
            # Call methods to generate plots for option prices, local volatility, and loss metrics
            self._plot_neural_option_price(t_tilde_, k_tilde_, T_nn_, K_nn_, phi_tilde_, grad_phi_t_tilde, dupire_eqn_error, step)
            self._plot_neural_local_volatility(t_tilde_, k_tilde_, T_nn_, K_nn_, sigma_nn_, sec_term, grad_phi_t_tilde, step)
            self._plot_losses(loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step)
            self._plot_losses_comparison(step)

        def _plot_neural_option_price(self, t_tilde_, k_tilde_, T_nn_, K_nn_, phi_tilde_, grad_phi_t_tilde, dupire_eqn_error, step):
            # Generate 3D plots for neural option prices, gradient, and Dupire equation error
            fig = plt.figure(figsize=[24, 8], dpi=450)

            # Neural option price surface plot
            phi_nn_ = S_0 * physics.neural_phi_tilde(t_tilde_, k_tilde_)

            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(phi_nn_, [N, m]), cmap=plt.cm.RdBu_r)
            ax1.set_xlabel('Strike price', labelpad=12, fontsize=16)
            ax1.set_ylabel('Maturity', labelpad=12, fontsize=16)
            ax1.set_zlabel('Neural option price', labelpad=12, fontsize=16)

            # Plot gradient of option price w.r.t. maturity
            ax_2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax_2.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(grad_phi_t_tilde, [N, m]), cmap=cm.inferno)
            ax_2.set_ylabel('Maturity: T', labelpad=12, fontsize=16)
            ax_2.set_xlabel('Strike price: K', labelpad=12, fontsize=16)
            ax_2.set_zlabel('grad_phi_t_tilde', labelpad=12, fontsize=16)

            # Plot Dupire equation error
            ax_3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax_3.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(dupire_eqn_error, [N, m]), cmap=cm.inferno)
            ax_3.set_ylabel('Maturity: T', labelpad=12, fontsize=16)
            ax_3.set_xlabel('Strike price: K', labelpad=12, fontsize=16)
            ax_3.set_zlabel('dupire_eqn_error', labelpad=12, fontsize=16)

            # Save the plot and display if `show` is enabled
            plt.savefig(os.path.join(dirname, f'phi_error_weight_{step}.png'))
            if show:
                plt.show()
            plt.close()

        def _plot_neural_local_volatility(self, t_tilde_, k_tilde_, T_nn_, K_nn_, sigma_nn_, sec_term, grad_phi_t_tilde, step):
            # Generate 3D plots for neural local volatility, second term, and weight surface
            fig = plt.figure(figsize=[24, 8], dpi=450)

            # Neural local volatility surface plot
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(sigma_nn_, [N, m]), cmap=plt.cm.RdBu_r)
            ax1.set_xlabel('Strike price K', labelpad=12, fontsize=16)
            ax1.set_ylabel('Maturity T', labelpad=12, fontsize=16)
            ax1.set_zlabel('Neural local volatility', labelpad=12, fontsize=16)

            # Plot second term in the Dupire equation
            ax_2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax_2.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(sec_term, [N, m]), cmap=cm.inferno)
            ax_2.set_ylabel('Maturity T', labelpad=12, fontsize=16)
            ax_2.set_xlabel('Strike price K', labelpad=12, fontsize=16)
            ax_2.set_zlabel('sec_term', labelpad=12, fontsize=16)

            # Plot weight surface for Dupire equation scaling
            weight_surf = physics.weight(grad_phi_t_tilde)

            ax_3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax_3.plot_surface(tf.reshape(K_nn_, [N, m]), tf.reshape(T_nn_, [N, m]), tf.reshape(weight_surf, [N, m]), cmap=cm.inferno)
            ax_3.set_ylabel('Maturity T', labelpad=12, fontsize=16)
            ax_3.set_xlabel('Strike price K', labelpad=12, fontsize=16)
            ax_3.set_zlabel('weight_surf', labelpad=12, fontsize=16)

            # Save and show plot
            plt.savefig(os.path.join(dirname, f'eta_error_weight_{step}.png'))
            if show:
                plt.show()
            plt.close()

        def _plot_losses(self, loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step):
            # Plot the loss metrics over time to track training progress
            fig, ax = plt.subplots(1, 2, figsize=[12, 2], dpi=450)

            # Plot primary losses in a log scale
            ax[0].semilogy(loss_phi_list, label='loss_phi')
            ax[0].semilogy(loss_dupire_list, label='loss_dupire')
            ax[0].semilogy(loss_reg_list, label='loss_reg')
            ax[0].legend(loc='upper right')

            # Plot relative error in volatility
            ax[1].plot(error_sigma_list, label='relative error sigma')
            ax[1].legend(loc='upper right')

            # Save and optionally display the plot
            plt.savefig(os.path.join(dirname, f'losses_{step}.png'))
            if show:
                plt.show()
            plt.close()

        def _plot_losses_comparison(self, step):
            # Compare neural predictions with exact references for option price and local volatility
            fig, ax = plt.subplots(1, 2, figsize=[12, 2], dpi=450)

            # Compare neural and exact option prices
            ax[0].plot(S_0 * phi_tilde_ref, label='Exact option price')
            ax[0].plot(S_0 * physics.neural_phi_tilde(t_tilde, k_tilde), label='Neural option price')
            ax[0].legend(loc='upper right')

            # Compare neural and reference local volatilities
            ax[1].plot(sigma_loc_ref, label='Ref local volatility', alpha=0.65)
            ax[1].plot(tf.sqrt(2 * physics.neural_eta_tilde(t_tilde, k_tilde) / t_max), label='Neural volatility')
            ax[1].legend(loc='upper right')

            # Save and show comparison plot
            plt.savefig(os.path.join(dirname, f'losses_comparison_{step}.png'))
            if show:
                plt.show()
            plt.close()

    plotter = Plotter()

    class Reprice:
        def __init__(self):
            super(Reprice, self).__init__()

        @tf.function
        def S_next_cal(self, t_now, S_now, dW_now):
            S_new = S_now + r * S_now * dt + physics.neural_sigma(t_now, S_now) * S_now * dW_now
            return S_new

        def run_mc(self, NN_phi_tilde=None, NN_eta_tilde=None):

            if NN_phi_tilde == None and NN_eta_tilde == None:
                NN_phi_tilde, NN_eta_tilde = processdata.load_nn(folder_name_load, iter_)

            S_0 = tf.reshape(tf.constant(S0, dtype=data_type),[-1,1])
            t_all = tf.cast(tf.reshape(np.linspace(0, N_t*dt, N_t), [-1,1]), dtype=data_type)
            S_list = [tf.cast(tf.reshape(np.full(M, S_0[0]), [1,M]), dtype=data_type)]
            dW_list = tf.cast(tf.concat([np.random.normal(0,1, size=[N_t,1]) * np.sqrt(dt) for i in range(M)], axis=1), dtype=data_type)

            time_0 = time.time()

            for i in range(N_t-1):
                t_now  = tf.repeat(t_all[i], M, axis=0)
                S_now  = S_list[-1]
                dW_now = dW_list[i]
                S_new = self.S_next_cal(t_now, S_now, dW_now)
                S_list.append(S_new)

            S_matrix = tf.concat(S_list, axis=0)
            
            time_1 = time.time()
            time_taken = time_1 - time_0
            
            return time_taken, t_all, S_matrix

        def plot_reprice_paths(self, time_taken, t_all, S_matrix, iter_):
            print(f'S_t obtained by solving local volatility SDE M = {M} times from t = [0, {N_t*dt}], computation time = {time_taken}')
            fig, ax = plt.subplots(figsize=[6, 2], dpi = 450)
            plt.plot(t_all, S_matrix[:,:1024], lw='0.1')
            plt.savefig(os.path.join(dirname,f'reprice_paths_{iter_}.png'))
            if show:
                plt.show()
            plt.close()

        def get_S_matrix(self, t_all, S_matrix):

            # Consider m_ strikes per maturity
            N_ = int(t_max*N_t) - int(t_min*N_t)
            m_ = 200

            T = tf.repeat(tf.reshape(t_all[int(t_min*N_t):int(t_max*N_t)], [-1,1]), m_, axis=1)
            K = tf.cast(tf.repeat(tf.reshape(np.linspace(k_min, k_max, m_), [1,-1]), len(T), axis=0), dtype=data_type)
            
            T_nn = tf.reshape(T, [-1,1])
            K_nn = tf.reshape(K, [-1,1])

            x = processdata.scale_data(T_nn, K_nn)
            t_tilde, k_tilde = x 

            S_T = S_matrix

            return S_T, T, K, T_nn, K_nn, N_, m_

        def phi_cal(self, S_T, T, K, N_, tensor=True):
            """
            compute option price per maturity-strike pair
            """
            if tensor:
                # Monte-Carlo estimation for the expectation
                E_ = tf.concat([tf.reshape(tf.reduce_mean(tf.nn.relu(tf.expand_dims(S_T[i], axis=0) -
                                                                     tf.expand_dims(K[i], axis=1)), axis=1), [1,-1]) for i in range(N_)], axis=0)
                phi_ = tf.exp(-r * T) * E_
            else:
                # Monte-Carlo estimation for the expectation
                phi_list_rec = []
                for i in range(len(T)):
                    phi_i = tf.exp(-r*T[i]) * tf.reduce_mean(tf.nn.relu(tf.reshape(S_T[i], [-1,1]) - tf.reshape(K[i], [1,-1])), axis=0, keepdims=True)
                    phi_list_rec.append(phi_i)
                phi_ = tf.reshape(tf.concat(phi_list_rec, axis=1), [-1,1])

            return phi_

        def make_plots(self, S_T, T, K, T_nn, K_nn, y_lim, z_lim, N_, m_, iter_):

            def _base_plot(val, val_label, savename, z_lim=None):
                fig = plt.figure(figsize=[8,6], dpi = 450)
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.plot_surface(tf.reshape(K, [N_,m_]), tf.reshape(T, [N_,m_]), tf.reshape(val, [N_,m_]), cmap = plt.cm.RdBu_r)
                ax1.tick_params(axis='x', labelsize=12)
                ax1.tick_params(axis='y', labelsize=12)
                ax1.tick_params(axis='z', labelsize=12)
                ax1.set_xlabel('Strike price', labelpad = 12, fontsize=16)
                ax1.set_ylabel('Maturity', labelpad = 12, fontsize=16)
                ax1.set_zlabel(val_label, labelpad = 12, fontsize=16)
                ax1.set_ylim(0, y_lim)
                if z_lim:
                    ax1.set_zlim(0, z_lim)
                plt.locator_params(nbins=5)
                plt.savefig(os.path.join(dirname,f'{savename}_{iter_}.png'))
                if show:
                    plt.show()
                plt.close()

            def _plots():
                print(f'neural phi from curve fitting')
                phi_fit_   = physics.neural_phi(T_nn, K_nn)
                _base_plot(phi_fit_, 'Exact Option Price', 'reprice_phi')
                print(f'neural sigma')
                sigma_nn_ = physics.neural_sigma(T_nn, K_nn)
                _base_plot(sigma_nn_, 'Volatility', 'reprice_eta', z_lim)
                print(f'recovered phi from local volatility')
                phi_rec_ = tf.reshape(self.phi_cal(S_T,T,K,N_), [-1,1])
                _base_plot(phi_rec_, 'Repriced Option Price', 'reprice_phi_rec')
                print(f'rel error')
                error_ = 100* (phi_fit_ - phi_rec_) / phi_fit_
                _base_plot(error_, 'Relative Error (%)', 'reprice_rel_err')

            _plots()

        def reprice_RMSE(self, S_matrix, N_):
            T_ref, K_ref, phi_ref, sigma_loc_ref, ids = processdata.read_csv(data_filename)

            # repricing RMSE on the grid of the market option price

            distinct_T = []

            for i in range(len(T_ref)):
                if T_ref[i] not in distinct_T:
                    distinct_T.append(T_ref[i])

            ids_list = []
            for T_i in distinct_T:
                ids,_ = np.where(T_ref == T_i)
                ids_list.append(ids)

            K_list_ref = []

            for ids_i in ids_list:
                K_i = tf.gather(K_ref, ids_i)
                K_list_ref.append(K_i)

            S_list_ref = []

            for T_i in distinct_T:
                S_i = S_matrix[int(T_i // dt)]
                S_list_ref.append(S_i)

            phi_rec = self.phi_cal(S_list_ref, distinct_T, K_list_ref, N_, False)
            
            rmse_rec = tf.sqrt(tf.reduce_mean(tf.square(phi_rec - phi_ref)))

            return rmse_rec

        def repricing(self, iter_, y_lim=1, z_lim=0.6, NN_phi_tilde=None, NN_eta_tilde=None, _plots=False):
            time_taken, t_all, S_matrix = self.run_mc(NN_phi_tilde, NN_eta_tilde)
            S_T, T, K, T_nn, K_nn, N_, m_ = self.get_S_matrix(t_all, S_matrix)
            rmse_rec = self.reprice_RMSE(S_T, N_)
            print ('Reprice RMSE: ', float(np.array(rmse_rec)))
            if _plots:
                self.plot_reprice_paths(time_taken, t_all, S_matrix, iter_)
                self.make_plots(S_T, T, K, T_nn, K_nn, y_lim, z_lim, N_, m_, iter_)

            return rmse_rec

    reprice = Reprice()

    class Trainer:
        def __init__(self):
            super(Trainer, self).__init__()
    
        def test(self):
            plotter.plot_res([], [], [], [], step=-1)

            time_0 = time.time()
            loss_phi = physics.loss_phi_cal()
            time_1 = time.time()

            print(f'loss_phi = {loss_phi}, computation time = {time_1 - time_0}')

            time_0 = time.time()
            loss_dupire, loss_reg = physics.loss_dupire_cal()
            time_1 = time.time()

            print(f'loss_dupire = {loss_dupire}, loss_reg = {loss_reg}, computation time = {time_1 - time_0}')

            physics.optimizer_NN_phi.learning_rate.assign(10**-4)
            physics.optimizer_NN_eta.learning_rate.assign(10**-4)

            lambda_pde = tf.constant(0.1, dtype=data_type)
            lambda_reg = tf.constant(0.1, dtype=data_type)

            time_0 = time.time()
            loss_phi, loss_dupire, loss_reg = physics.train_step(lambda_pde, lambda_reg)
            time_1 = time.time()

            print(f'computation time = {time_1 - time_0}')
            print(f'loss_phi = {loss_phi}, loss_dupire = {loss_dupire}, loss_reg = {loss_reg}')
            print()

            time_0 = time.time()
            loss_phi, loss_dupire, loss_reg = physics.train_step(lambda_pde, lambda_reg)
            time_1 = time.time()

            print(f'computation time = {time_1 - time_0}')
            print(f'loss_phi = {loss_phi}, loss_dupire = {loss_dupire}, loss_reg = {loss_reg}')
            print()

        def run(self):

            learning_rate_ = lr

            physics.optimizer_NN_phi.learning_rate.assign(learning_rate_)
            physics.optimizer_NN_eta.learning_rate.assign(learning_rate_/div_lr)

            loss_phi_list     = []
            loss_dupire_list  = []
            loss_reg_list     = []

            error_sigma_list  = []
            rmse_sigma_list   = []

            lambda_pde = tf.constant(ldup, dtype=data_type)
            lambda_reg = tf.constant(1.0, dtype=data_type)

            for iter_ in range(num_epochs+1):

                loss_phi, loss_dupire, loss_reg = physics.train_step(lambda_pde, lambda_reg)

                loss_phi_list.append(loss_phi)
                loss_dupire_list.append(loss_dupire)
                loss_reg_list.append(loss_reg)

                # compute the relative error of neural local volatility
                sigma_exact_ = sigma_loc_ref
                sigma_nn_ = tf.sqrt(2*physics.neural_eta_tilde(t_tilde, k_tilde)/(t_max))
                error_sigma = tf.reduce_mean(tf.abs(sigma_exact_ - sigma_nn_) / sigma_exact_)
                rmse_sigma  = tf.sqrt(tf.reduce_mean(tf.square(1 - sigma_nn_ / sigma_exact_)))
                error_sigma_list.append(error_sigma)
                rmse_sigma_list.append(rmse_sigma)

                if iter_ % print_epochs ==0:
                    rmse_fit = tf.sqrt(tf.reduce_mean(tf.square(physics.neural_phi(T_nn, K_nn) - phi_ref)))
                    print(f'iter = {iter_}, lambda = {lambda_pde}: loss_phi = {loss_phi_list[-1]}, loss_dupire = {loss_dupire_list[-1]}, error_sigma = {error_sigma_list[-1]}, rmse_fit = {rmse_fit}')

                if iter_ % 2000 == 0 and iter_ != 0:

                    learning_rate_ /= 1.1

                    physics.optimizer_NN_phi.learning_rate.assign(learning_rate_)
                    physics.optimizer_NN_eta.learning_rate.assign(learning_rate_)

                if iter_ % save_epochs == 0 and iter_ != 0:
                    plotter.plot_res(loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, iter_)
                    if iter_ > int(num_epochs-1):
                        processdata.save_nn(physics.NN_phi_tilde, physics.NN_eta_tilde, iter_)
                        reprice_ = reprice.repricing(iter_=iter_, NN_phi_tilde=physics.NN_phi_tilde, NN_eta_tilde=physics.NN_eta_tilde)
                        repricings.append([iter_, float(np.array(reprice_))])

            return rmse_sigma_list, error_sigma_list

        def make_plots(self, rmse_sigma_list, error_sigma_list):

            print(f'relative rmse at the end of the training = {rmse_sigma_list[-1] }')
            print(f'smallest relative rmse during the training = {tf.reduce_min(rmse_sigma_list) }')

            print(f'relative error at the end of the training = {error_sigma_list[-1] }')
            print(f'smallest relative error during the training = {tf.reduce_min(error_sigma_list) }')

            # save model at the end of the training
            processdata.save_nn(physics.NN_phi_tilde, physics.NN_eta_tilde, 'final')

            fig = plt.figure(dpi=450)
            plt.semilogy(rmse_sigma_list, label='rmse')
            plt.semilogy(error_sigma_list, label='error')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(dirname,f'errors_final.png'))
            if show:
                plt.show()
            plt.close()

    trainer = Trainer()

    trainer.test()
    rmse_sigma_list, error_sigma_list = trainer.run()
    trainer.make_plots(rmse_sigma_list, error_sigma_list)
    
    reprice_ = reprice.repricing(iter_='final', NN_phi_tilde=physics.NN_phi_tilde, NN_eta_tilde=physics.NN_eta_tilde, _plots=True)
    repricings.append(['final', float(np.array(reprice_))])
    all_repricings.append([date_S0, ldup, k_min, k_max, repricings])

for ldup in ldups:
    for date_S0 in dates_S0:
        main(ldup, date_S0)

print (all_repricings)