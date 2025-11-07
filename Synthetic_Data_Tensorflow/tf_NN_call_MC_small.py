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

ldups = [1,0,0.5,1.5,2]

#NN Params
gaussian_phi = 0.5
gaussian_eta = 0.5
num_res_blocks = 3
div_lr = 10
lr = 10**-3

#Plotting params
N_ = 256
m_ = 256

#MC Params
#samples
N,m = 3,6
#spot price 
S0 = 1000
#risk free rate
r_ = 0.04
# number of samples
M  = 10**4
# number of time steps
N_t = 1000
# size of time step
dt = 10**-3

def main(ldup):

    print (ldup)
    print ('')

    identifier = f'batch_call_MC_ldup_{ldup}'

    folder_name_load = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    folder_name_save = f'{identifier}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    dirname = folder_name_save

    S_0 = tf.constant(S0, dtype=data_type)
    r = tf.constant(r_,    dtype=data_type)

    class LoadData:
        def __init__(self):
            super(LoadData, self).__init__()
            self.N = N
            self.m = m
            self.N_t = 150
            self.M = 10**6

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

        def exact_sigma(self, t, x):
            """
            define local volatility, which is a function of t and x
            """

            y_ = tf.sqrt(x + 0.1) * (t + 0.1)

            sigma_ = 0.3 + y_ * tf.exp(-y_)

            return sigma_

        def plot_loc_vol(self):

        	t_ = np.linspace(0, 1,  1024)
        	x_ = np.linspace(0.5, 3, 1024)
        	t_mesh, x_mesh = np.meshgrid(t_, x_)
        	sigma_surf = np.array(exact_sigma(np.ravel(t_mesh), np.ravel(x_mesh)))
        	sigma_surf_mesh = sigma_surf.reshape(x_mesh.shape)

        	fig = plt.figure(figsize=[8,6], dpi = 450)
        	ax = fig.add_subplot(111, projection='3d')
        	ax.plot_surface(t_mesh, x_mesh, sigma_surf_mesh)
        	ax.set_xlabel('t')
        	ax.set_ylabel('x')
        	ax.set_zlabel('local volatility')
        	plt.savefig(dirname, 'loc_vol.png')
        	if show:
        		plt.show()
        	else:
        		plt.close()

        def run_mc(self):

        	# dimension of the random variable
        	d  = 1
        	# number of samples
        	M  = self.M
        	# number of time steps
        	N_t  = self.N_t
        	# size of time step
        	dt = 0.01

        	S_0 = tf.reshape(tf.constant([1000.0], dtype=data_type), [-1,1])

        	t_all = tf.cast(tf.reshape(np.linspace(0, N_t*dt, N_t), [-1,1]), dtype=data_type)

        	S_list = [tf.cast(tf.reshape(np.full(M, S_0[0]), [1,M]), dtype=data_type)]

        	dW_list = tf.cast(tf.concat([np.random.normal(0,1, size=[N_t,1]) * np.sqrt(dt) for i in range(M)], axis=1), dtype=data_type)

        	for i in range(N_t-1):
        	    t_now = t_all[i]
        	    S_now = S_list[-1]

        	    S_new = S_now + r * S_now * dt + self.exact_sigma(t_now, S_now/S_0) * S_now * dW_list[i]

        	    S_list.append(S_new)

        	S_matrix = tf.concat(S_list, axis=0)

        	print(f'S_t obtained by solving local volatility SDE M = {M} times from t = [0, {N_t*dt}]')

        	return S_matrix, t_all

        def get_T_K(self):
        	# Consider m strikes per maturity
        	N = self.N
        	m = self.m
        	N_t = self.N_t

        	S_matrix, t_all = self.run_mc()

        	indices = tf.cast(tf.linspace(30, N_t-1, N), tf.int32)
        	t_all_T = tf.gather(t_all, indices)
        	S_T = tf.gather(S_matrix, indices, axis=0)

        	T = tf.repeat(tf.reshape(t_all_T, [-1,1]), m, axis=1)
        	K = tf.cast(tf.repeat(tf.reshape(np.linspace(500, 3000, m), [1,-1]), len(T), axis=0), dtype=data_type)

        	def exact_phi(T, K):
        	    """
        	    compute option price per maturity-strike pair
        	    """

        	    # Monte-Carlo estiation for the expectation
        	    E_ = tf.concat([tf.reshape(tf.reduce_mean(tf.nn.relu(tf.expand_dims(S_T[i], axis=0) - tf.expand_dims(K[i], axis=1)), axis=1), [1,-1]) for i in range(N)], axis=0)

        	    phi_ = tf.exp(-r * T) * E_

        	    return phi_

        	# compute option price at given (T, K)
        	phi_exact = exact_phi(T,K)

        	T_nn = tf.reshape(T, [-1,1])
        	K_nn = tf.reshape(K, [-1,1])
        	phi_ref = tf.reshape(phi_exact, [-1,1])

        	self.save_phi_exact(phi_ref)

        	return T_nn, K_nn, phi_ref, S_matrix, t_all

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

        def make_prelim_plots(self, S_matrix, t_all, t_tilde, k_tilde, phi_tilde_ref):
            N = self.N
            m = self.m

            fig, ax = plt.subplots(figsize=[12, 3], dpi = 450)
            plt.plot(t_all, S_matrix[:,:1024], lw='0.1')
            plt.savefig(os.path.join(dirname, f'mc_paths.png'))
            if show:
            	plt.show()
            else:
            	plt.close()

            fig = plt.figure(figsize=[14,6], dpi = 450)
            ax = fig.add_subplot(121)
            ax.scatter(t_tilde, k_tilde, s=0.1)
            ax.set_xlabel('t_tilde')
            ax.set_ylabel('k_tilde')
            ax = fig.add_subplot(122, projection='3d')
            ax.plot_surface(tf.reshape(k_tilde, [N,m]), tf.reshape(t_tilde, [N,m]), tf.reshape(phi_tilde_ref, [N,m]))
            ax.set_ylabel('Scaled maturity: t_tilde')
            ax.set_xlabel('Scaled strike price: k_tilde')
            ax.set_zlabel('Scaled option price: phi_tilde')
            plt.savefig(os.path.join(dirname, f'scaled_t_k.png'))
            if show:
            	plt.show()
            else:
            	plt.close()

    processdata = LoadData()

    T_nn, K_nn, phi_ref, S_matrix, t_all = processdata.get_T_K()
    phi_tilde_ref = phi_ref / S_0

    t_min, t_max, k_min, k_max = processdata.get_min_max(T_nn, K_nn)
    x = processdata.scale_data(T_nn, K_nn)
    t_tilde, k_tilde = x
    x_random = processdata.get_min_max_random(t_tilde, k_tilde, T_nn)
    [t_min_random, t_max_random, k_min_random, k_max_random] = x_random
    processdata.make_prelim_plots(S_matrix, t_all, t_tilde, k_tilde, phi_tilde_ref)

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

        def exact_eta_tilde(self, t_tilde, k_tilde):
            T_ = t_max * t_tilde
            K_ = tf.exp(r*T_) * k_max * k_tilde / S_0
            eta_tilde_ = 0.5 * t_max * processdata.exact_sigma(T_, K_)**2
            return eta_tilde_

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
            T_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(T_min, T_max, N_), dtype=data_type), [-1, 1]), m_, axis=1), [-1, 1])
            K_ = tf.reshape(tf.repeat(tf.reshape(tf.cast(np.linspace(K_min, K_max, m_), dtype=data_type), [1, -1]), N_, axis=0), [-1, 1])

            # Reshape tensors for easier manipulation
            T_nn_ = tf.reshape(T_, [-1, 1])
            K_nn_ = tf.reshape(K_, [-1, 1])

            x_ = processdata.scale_data(T_nn_, K_nn_)
            t_tilde_, k_tilde_ = x_ 

            # Obtain model predictions for option prices and local volatility
            phi_tilde_ = physics.neural_phi_tilde(t_tilde_, k_tilde_)
            sigma_nn_ = tf.sqrt(2 * physics.neural_eta_tilde(t_tilde_, k_tilde_) / t_max)
            sigma_tilde_ref_ = tf.sqrt(2*physics.exact_eta_tilde(t_tilde_, k_tilde_) / (t_max))

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
            weight_surf_phi_ref = physics.weight(phi_tilde_ref)
            weight_surf_gradt = physics.weight(grad_phi_t_tilde)
            err_surf_phi = (tf.reshape(phi_tilde_ref, [N,m]) - tf.reshape(physics.neural_phi_tilde(t_tilde, k_tilde), [N,m]))
            err_surf_sigma = (tf.reshape(sigma_tilde_ref_, [N_,m_]) - tf.reshape(sigma_nn_, [N_,m_]))

            ########################################################################################################
            # Call methods to generate plots for option prices, local volatility, and loss metrics
            self._plot_neural_option_price(T_nn_, K_nn_, phi_tilde_, err_surf_phi, err_surf_phi, step)
            self._plot_neural_local_volatility(T_nn_, K_nn_, sigma_nn_, sigma_tilde_ref_, err_surf_sigma, step)
            self._plot_neural_gradients_phi(T_nn_, K_nn_, sec_term, grad_phi_t_tilde, dupire_eqn_error, weight_surf_gradt, step)
            self._plot_losses(loss_phi_list, loss_dupire_list, loss_reg_list, error_sigma_list, step)

        def _plot_neural_option_price(self, T_nn_, K_nn_, phi_tilde_, err_surf_phi, weight_surf_phi_ref, step):
            # Generate 3D plots for neural option prices, exact option prices, error and weights
            fig = plt.figure(figsize=[24, 8], dpi=450)

            # Neural option price surface plot
            ax_1 = fig.add_subplot(1,4,1, projection='3d')
            ax_1.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(S_0*phi_tilde_, [N_,m_]), cmap=cm.RdBu_r,linewidth=0)
            ax_1.set_ylabel('Maturity: t_tilde')
            ax_1.set_xlabel('Strike price: k_tilde')
            ax_1.set_zlabel('Neural option price: phi_tilde')

            ax_2 = fig.add_subplot(1,4,2, projection='3d')
            ax_2.plot_surface(tf.reshape(K_nn, [N,m]), tf.reshape(T_nn, [N,m]), tf.reshape(S_0*phi_tilde_ref, [N,m]), cmap=cm.RdBu_r,linewidth=0)
            ax_2.set_ylabel('Maturity: t_tilde')
            ax_2.set_xlabel('Strike price: k_tilde')
            ax_2.set_zlabel('Exact option price: phi_tilde')

            ax_3 = fig.add_subplot(1,4,3, projection='3d')
            ax_3.plot_surface(tf.reshape(K_nn, [N,m]), tf.reshape(T_nn, [N,m]), tf.reshape(S_0*err_surf_phi, [N,m]), cmap=cm.RdBu_r,linewidth=0)
            ax_3.set_ylabel('Maturity: t_tilde')
            ax_3.set_xlabel('Strike price: k_tilde')
            ax_3.set_zlabel('Error phi_tilde')

            ax_4 = fig.add_subplot(1,4,4, projection='3d')
            ax_4.plot_surface(tf.reshape(K_nn, [N,m]), tf.reshape(T_nn, [N,m]), tf.reshape(weight_surf_phi_ref, [N,m]), cmap=cm.inferno,linewidth=0)
            ax_4.set_ylabel('Maturity: t_tilde')
            ax_4.set_xlabel('Strike price: k_tilde')
            ax_4.set_zlabel('weight_surf ')
            plt.savefig(os.path.join(dirname, f'phi_{step}.png'))
            if show:
                plt.show()
            else:
                plt.close()

        def _plot_neural_local_volatility(self, T_nn_, K_nn_, sigma_nn_, sigma_tilde_ref_, err_surf_sigma, step):
            # Generate 3D plots for neural local volatility, exact local volatility, and error
            fig = plt.figure(figsize=[24, 8], dpi=450)

            # Neural local volatility surface plot
            ax_1 = fig.add_subplot(1,3,1, projection='3d')
            ax_1.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(sigma_nn_, [N_,m_]), cmap=cm.RdBu_r,linewidth=0)
            ax_1.set_ylabel('Maturity: t_tilde')
            ax_1.set_xlabel('Strike price: k_tilde')
            ax_1.set_zlabel('Neural volatility: sigma_tilde')

            ax_2 = fig.add_subplot(1,3,2, projection='3d')
            ax_2.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(sigma_tilde_ref_, [N_,m_]), cmap=cm.RdBu_r,linewidth=0)
            ax_2.set_ylabel('Maturity: t_tilde')
            ax_2.set_xlabel('Strike price: k_tilde')
            ax_2.set_zlabel('Exact volatility: sigma_tilde')

            ax_3 = fig.add_subplot(1,3,3, projection='3d')
            ax_3.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(err_surf_sigma, [N_,m_]), cmap=cm.RdBu_r,linewidth=0)
            ax_3.set_ylabel('Maturity: t_tilde')
            ax_3.set_xlabel('Strike price: k_tilde')
            ax_3.set_zlabel('Error sigma_tilde')

            # Save and show plot
            plt.savefig(os.path.join(dirname, f'eta_error_weight_{step}.png'))
            if show:
                plt.show()
            plt.close()

        def _plot_neural_gradients_phi(self, T_nn_, K_nn_, sec_term, grad_phi_t_tilde, dupire_eqn_error, weight_surf_gradt, step):
            # Generate 3D plots for neural gradients
            fig = plt.figure(figsize=[24,8], dpi = 450)

            ax_1 = fig.add_subplot(1,4,1, projection='3d')
            ax_1.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(grad_phi_t_tilde, [N_,m_]), cmap=cm.inferno,linewidth=0)
            ax_1.set_ylabel('Maturity: t_tilde')
            ax_1.set_xlabel('Strike price: k_tilde')
            ax_1.set_zlabel('grad_phi_t_tilde')

            ax_2 = fig.add_subplot(1,4,2, projection='3d')
            ax_2.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(sec_term, [N_,m_]), cmap=cm.inferno,linewidth=0)
            ax_2.set_ylabel('Maturity: t_tilde')
            ax_2.set_xlabel('Strike price: k_tilde')
            ax_2.set_zlabel('sec_term')

            ax_3 = fig.add_subplot(1,4,3, projection='3d')
            ax_3.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(dupire_eqn_error, [N_,m_]), cmap=cm.inferno,linewidth=0)
            ax_3.set_ylabel('Maturity: t_tilde')
            ax_3.set_xlabel('Strike price: k_tilde')
            ax_3.set_zlabel('dupire_eqn_error')

            ax_4 = fig.add_subplot(1,4,4, projection='3d')
            ax_4.plot_surface(tf.reshape(K_nn_, [N_,m_]), tf.reshape(T_nn_, [N_,m_]), tf.reshape(weight_surf_gradt , [N_,m_]), cmap=cm.inferno,linewidth=0)
            ax_4.set_ylabel('Maturity: t_tilde')
            ax_4.set_xlabel('Strike price: k_tilde')
            ax_4.set_zlabel('weight_surf ')

            # Save and optionally display the plot
            plt.savefig(os.path.join(dirname, f'gradphi_{step}.png'))
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

    plotter = Plotter()

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
                sigma_exact_ = tf.sqrt(2*physics.exact_eta_tilde(t_tilde, k_tilde)/(t_max))
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

for ldup in ldups:
    main(ldup)