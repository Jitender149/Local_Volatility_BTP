from NN import *

class NNTrain:
    def __init__(self, MonteCarloLocalVolatility, ScaleQuantities, nn_params, learning_rate, boundary_epsilon, lambda_fit, lambda_pde, lambda_arb, lambda_reg, lambda_ini, num_epochs, print_epochs, save_epochs, dirname, scheduler=None):
        
        self.K = ScaleQuantities.K 
        self.T = ScaleQuantities.T
        self.t_tilde = ScaleQuantities.t_tilde
        self.k_tilde = ScaleQuantities.k_tilde

        self.t_min = ScaleQuantities.t_min
        self.t_max = ScaleQuantities.t_max
        self.k_min = ScaleQuantities.k_min
        self.k_max = ScaleQuantities.k_max

        self.r = ScaleQuantities.r
        self.phi_tilde_ref = ScaleQuantities.phi_tilde_ref

        self.exact_sigma_torch = MonteCarloLocalVolatility.exact_sigma_torch

        self.N = ScaleQuantities.N
        self.m = ScaleQuantities.m
        self.S_0 = ScaleQuantities.S_0
        self.data_type = ScaleQuantities.data_type
        self.show = ScaleQuantities.show 
        self.device = ScaleQuantities.device

        self.scaling_function = ScaleQuantities.scaling_function
        self.inverse_scaling_function = ScaleQuantities.inverse_scaling_function

        self.nn_params = nn_params
        self.NN_phi_tilde = NetPhiTilde(self.nn_params).to(self.device)
        self.NN_eta_tilde = NetEtaTilde(self.nn_params).to(self.device)

        self.learning_rate = learning_rate
        self.optimizer_NN_phi = optim.Adam(self.NN_phi_tilde.parameters(), lr=self.learning_rate)
        self.optimizer_NN_eta = optim.Adam(self.NN_eta_tilde.parameters(), lr=self.learning_rate / 4)

        if scheduler is None:
            self.scheduler_NN_phi = LambdaLR(self.optimizer_NN_phi, lr_lambda=lambda iter_: self.lr_lambda(iter_))
            self.scheduler_NN_eta = LambdaLR(self.optimizer_NN_eta, lr_lambda=lambda iter_: self.lr_lambda(iter_) / 4)
        else:
            self.scheduler_NN_phi = scheduler[0](self.optimizer_NN_phi, **scheduler[1])
            self.scheduler_NN_eta = scheduler[0](self.optimizer_NN_eta, **scheduler[1])

        self.boundary_epsilon = boundary_epsilon

        self.lambda_fit = lambda_fit
        self.lambda_ini = lambda_ini
        self.lambda_pde = lambda_pde
        self.lambda_arb = lambda_arb
        self.lambda_reg = lambda_reg

        self.num_epochs = num_epochs
        self.print_epochs = print_epochs
        self.save_epochs = save_epochs
        self.dirname = dirname

    def lr_lambda(self, iter_):
        if iter_ == 0:
            return 1.0  # Keep the initial learning rate
        else:
            return 1 / (1.2 ** (iter_ // 2000))

    def neural_phi_tilde(self, t_tilde, k_tilde):
        input_concat = torch.cat([t_tilde, k_tilde], dim=1)
        phi_nn_ = self.NN_phi_tilde(input_concat)
        return phi_nn_

    def neural_eta_tilde(self, t_tilde, k_tilde):
        input_concat = torch.cat([t_tilde, k_tilde], dim=1)
        eta_nn_ = self.NN_eta_tilde(input_concat)
        return eta_nn_

    def exact_eta_tilde(self, t_tilde, k_tilde):

        T_ = self.inverse_scaling_function(t_tilde, self.t_min, self.t_max)
        K_ = torch.exp(self.r * T_) * self.inverse_scaling_function(k_tilde, self.k_min, self.k_max)

        eta_tilde_ = 0.5 * (self.t_max - self.t_min) * self.exact_sigma_torch(T_, K_)**2
        return eta_tilde_

    def loss_phi_cal(self):
        phi_tilde_nn = self.neural_phi_tilde(self.t_tilde, self.k_tilde)
        with torch.no_grad():
            weight_ = torch.clip(torch.mean(torch.abs(self.phi_tilde_ref)) / torch.abs(self.phi_tilde_ref), min=0.1, max=10)
        loss_phi_ = torch.mean(weight_ * (phi_tilde_nn - self.phi_tilde_ref) ** 2)
        return loss_phi_

    def loss_dupire_cal(self):
        t_tilde_0 = torch.full((self.N, 1), 0, dtype=self.data_type)
        t_tilde_1 = torch.full((self.N, 1), 0, dtype=self.data_type)
        k_tilde_0 = torch.full((self.m, 1), 1, dtype=self.data_type)
        k_tilde_1 = torch.full((self.m, 1), 1, dtype=self.data_type)

        t_tilde_bulk_random = torch.rand((128**2, 1), dtype=self.data_type)
        k_tilde_bulk_random = torch.rand((128**2, 1), dtype=self.data_type)

        t_tilde_random = torch.cat([t_tilde_0, t_tilde_1, t_tilde_bulk_random], dim=0).to(self.device)
        k_tilde_random = torch.cat([k_tilde_0, k_tilde_1, k_tilde_bulk_random], dim=0).to(self.device)

        t_tilde_random.requires_grad = True
        k_tilde_random.requires_grad = True

        phi_tilde_ = self.neural_phi_tilde(t_tilde_random, k_tilde_random)

        grad_phi_t_tilde = torch.autograd.grad(phi_tilde_, t_tilde_random, grad_outputs=torch.ones_like(phi_tilde_), create_graph=True)[0]
        grad_phi_k_tilde = torch.autograd.grad(phi_tilde_, k_tilde_random, grad_outputs=torch.ones_like(phi_tilde_), create_graph=True)[0]

        grad_phi_kk_tilde = torch.autograd.grad(grad_phi_k_tilde, k_tilde_random, grad_outputs=torch.ones_like(grad_phi_k_tilde), create_graph=True)[0]

        eta_tilde_ = self.neural_eta_tilde(t_tilde_random, k_tilde_random)

        dupire_eqn = grad_phi_t_tilde - eta_tilde_ * self.inverse_scaling_function(k_tilde_random, self.k_min, self.k_max, True)**2 * grad_phi_kk_tilde
        inferred_eta = grad_phi_t_tilde / (self.inverse_scaling_function(k_tilde_random, self.k_min, self.k_max, True)**2 * grad_phi_kk_tilde)

        with torch.no_grad():
            weight_ = torch.clamp(torch.mean(torch.abs(grad_phi_t_tilde)) / torch.abs(grad_phi_t_tilde), min=0.1, max=10)

        loss_dupire_ = torch.mean(weight_ * dupire_eqn**2)
        loss_reg_ = torch.mean(weight_ * torch.relu(-grad_phi_t_tilde * grad_phi_kk_tilde))
        loss_arb_ = torch.mean(weight_ * torch.relu(-(grad_phi_t_tilde - self.r * self.inverse_scaling_function(k_tilde_random, self.k_min, self.k_max, True) * grad_phi_k_tilde)))

        return loss_dupire_, loss_reg_, loss_arb_

    def loss_boundary_cal(self):
        t_tilde_random = torch.full((128, 1), 0, dtype=self.data_type).to(self.device)
        k_tilde_random = torch.rand((128, 1), dtype=self.data_type).to(self.device)*self.boundary_epsilon
        exact_phi_t_0 = torch.maximum(
            1 - self.inverse_scaling_function(k_tilde_random, self.k_min, self.k_max) / self.S_0,
            torch.tensor(0.0, dtype=self.data_type)
        )
        phi_tilde_ = self.neural_phi_tilde(t_tilde_random, k_tilde_random)
        with torch.no_grad():
            weight_ = torch.clamp(torch.mean(torch.abs(exact_phi_t_0)) / torch.abs(exact_phi_t_0), min=0.1, max=10)

        loss_boundary = torch.mean(weight_ * (phi_tilde_ - exact_phi_t_0)**2)
        return loss_boundary

    def train_step(self):
        self.optimizer_NN_phi.zero_grad()
        self.optimizer_NN_eta.zero_grad()
        loss_dupire, loss_reg, loss_arb = self.loss_dupire_cal()
        loss_dupire.backward()  # Only need the gradients for the dupire loss
        self.optimizer_NN_eta.step()  # Update NN_eta_tilde's parameters
        self.scheduler_NN_eta.step()
            
        self.optimizer_NN_phi.zero_grad()
        self.optimizer_NN_eta.zero_grad()
        loss_phi = self.loss_phi_cal()
        loss_dupire, loss_reg, loss_arb = self.loss_dupire_cal()
        loss_boundary = self.loss_boundary_cal()
        loss_total = self.lambda_fit * loss_phi + self.lambda_pde * loss_dupire + self.lambda_arb * loss_arb + self.lambda_reg * loss_reg + self.lambda_ini * loss_boundary
        loss_total.backward()
        self.optimizer_NN_phi.step()
        self.scheduler_NN_phi.step()

        return loss_phi.item(), loss_dupire.item(), loss_reg.item(), loss_boundary.item(), loss_arb.item()

    def plot_res(self, loss_phi_list, loss_dupire_list, loss_reg_list, loss_boundary_list, loss_arb_list, error_sigma_list, step):

        t_tilde = self.t_tilde
        k_tilde = self.k_tilde
        K = self.K 
        T = self.T 
        phi_tilde_ref = self.phi_tilde_ref
        N = self.N 
        m = self.m 
        k_min = self.k_min
        k_max = self.k_max
        t_min = self.t_min
        t_max = self.t_max
        r = self.r
        show = self.show

        phi_tilde = self.neural_phi_tilde(t_tilde, k_tilde)
        sigma_tilde = torch.sqrt(self.neural_eta_tilde(t_tilde, k_tilde))
        sigma_tilde_ref = torch.sqrt(self.exact_eta_tilde(t_tilde, k_tilde))

        fig = plt.figure(figsize=[24,8], dpi = 450)

        ax_1 = fig.add_subplot(1,4,1, projection='3d')
        ax_1.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(phi_tilde, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_1.set_ylabel('Maturity: t_tilde')
        ax_1.set_xlabel('Strike price: k_tilde')
        ax_1.set_zlabel('Neural option price: phi_tilde')

        ax_2 = fig.add_subplot(1,4,2, projection='3d')
        ax_2.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(phi_tilde_ref, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_2.set_ylabel('Maturity: t_tilde')
        ax_2.set_xlabel('Strike price: k_tilde')
        ax_2.set_zlabel('Exact option price: phi_tilde')

        err_surf_phi = torch.reshape(phi_tilde_ref, [N,m]) - torch.reshape(phi_tilde, [N,m])

        ax_3 = fig.add_subplot(1,4,3, projection='3d')
        ax_3.plot_surface(K.cpu().numpy(), T.cpu().numpy(), torch.reshape(err_surf_phi, [len(T),m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_3.set_ylabel('Maturity: t_tilde')
        ax_3.set_xlabel('Strike price: k_tilde')
        ax_3.set_zlabel('Error phi_tilde')

        with torch.no_grad():
            weight_surf = torch.clamp(
                torch.mean(torch.abs(phi_tilde_ref)) / torch.abs(phi_tilde_ref),
                min=0.1, max=10
            )

        ax_4 = fig.add_subplot(1,4,4, projection='3d')
        ax_4.plot_surface(K.detach().cpu().numpy(), T.detach().cpu().numpy(), torch.reshape(weight_surf , [len(T),m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_4.set_ylabel('Maturity: t_tilde')
        ax_4.set_xlabel('Strike price: k_tilde')
        ax_4.set_zlabel('weight_surf ')

        plt.savefig(os.path.join(self.dirname,f'phi_error_weight_{step}.png'))
        if show:
            plt.show()
        plt.close()

        t_tilde.requires_grad = True
        k_tilde.requires_grad = True

        phi_tilde_ = self.neural_phi_tilde(t_tilde, k_tilde)

        grad_phi_t_tilde = torch.autograd.grad(outputs=phi_tilde_, inputs=t_tilde,
                                               grad_outputs=torch.ones_like(phi_tilde_),
                                               create_graph=True)[0]

        grad_phi_k_tilde = torch.autograd.grad(outputs=phi_tilde_, inputs=k_tilde,
                                               grad_outputs=torch.ones_like(phi_tilde_),
                                               create_graph=True)[0]

        grad_phi_kk_tilde = torch.autograd.grad(outputs=grad_phi_k_tilde, inputs=k_tilde,
                                                grad_outputs=torch.ones_like(grad_phi_k_tilde),
                                                create_graph=True)[0]

        inferred_eta = grad_phi_t_tilde / (self.inverse_scaling_function(k_tilde, self.k_min, self.k_max, True)**2 * grad_phi_kk_tilde)
        inferred_sigma = torch.sqrt(inferred_eta)

        eta_tilde_ = self.neural_eta_tilde(t_tilde, k_tilde)
        sec_term = eta_tilde_ * self.inverse_scaling_function(k_tilde, self.k_min, self.k_max, True)**2 * grad_phi_kk_tilde

        dupire_eqn_error = grad_phi_t_tilde - sec_term

        fig = plt.figure(figsize=[24,8], dpi = 450)

        ax_1 = fig.add_subplot(1,4,1, projection='3d')
        ax_1.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(grad_phi_t_tilde, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_1.set_ylabel('Maturity: t_tilde')
        ax_1.set_xlabel('Strike price: k_tilde')
        ax_1.set_zlabel('grad_phi_t_tilde')

        ax_2 = fig.add_subplot(1,4,2, projection='3d')
        ax_2.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(sec_term, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_2.set_ylabel('Maturity: t_tilde')
        ax_2.set_xlabel('Strike price: k_tilde')
        ax_2.set_zlabel('sec_term')

        ax_3 = fig.add_subplot(1,4,3, projection='3d')
        ax_3.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(dupire_eqn_error, [len(T),m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_3.set_ylabel('Maturity: t_tilde')
        ax_3.set_xlabel('Strike price: k_tilde')
        ax_3.set_zlabel('dupire_eqn_error')

        # Compute weight_surf in PyTorch
        with torch.no_grad():  # This is equivalent to stop_gradient in TensorFlow
            weight_surf = torch.clamp(
                torch.mean(torch.abs(grad_phi_t_tilde)) / torch.abs(grad_phi_t_tilde),
                min=0.1, max=10
            )

        ax_4 = fig.add_subplot(1,4,4, projection='3d')
        ax_4.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(weight_surf , [len(T),m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_4.set_ylabel('Maturity: t_tilde')
        ax_4.set_xlabel('Strike price: k_tilde')
        ax_4.set_zlabel('weight_surf ')

        plt.savefig(os.path.join(self.dirname,f'dupire_error_weight_{step}.png'))
        if show:
            plt.show()
        plt.close()

        fig = plt.figure(figsize=[24,8], dpi = 450)

        ax_1 = fig.add_subplot(1,3,1, projection='3d')
        ax_1.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(sigma_tilde, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_1.set_ylabel('Maturity: t_tilde')
        ax_1.set_xlabel('Strike price: k_tilde')
        ax_1.set_zlabel('Neural volatility: sigma_tilde')

        ax_2 = fig.add_subplot(1,3,2, projection='3d')
        ax_2.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(sigma_tilde_ref, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_2.set_ylabel('Maturity: t_tilde')
        ax_2.set_xlabel('Strike price: k_tilde')
        ax_2.set_zlabel('Exact volatility: sigma_tilde')

        err_surf_sigma = torch.reshape(sigma_tilde_ref, [N,m]) - torch.reshape(sigma_tilde, [N,m])

        ax_3 = fig.add_subplot(1,3,3, projection='3d')
        ax_3.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(err_surf_sigma, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_3.set_ylabel('Maturity: t_tilde')
        ax_3.set_xlabel('Strike price: k_tilde')
        ax_3.set_zlabel('Error sigma_tilde')

        plt.savefig(os.path.join(self.dirname,f'sigma_error_{step}.png'))
        if show:
            plt.show()
        plt.close()

        fig = plt.figure(figsize=[24,8], dpi = 450)

        ax_1 = fig.add_subplot(1,3,1, projection='3d')
        ax_1.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(inferred_sigma, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_1.set_ylabel('Maturity: t_tilde')
        ax_1.set_xlabel('Strike price: k_tilde')
        ax_1.set_zlabel('Inferred volatility: sigma_tilde')

        ax_2 = fig.add_subplot(1,3,2, projection='3d')
        ax_2.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(sigma_tilde_ref, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_2.set_ylabel('Maturity: t_tilde')
        ax_2.set_xlabel('Strike price: k_tilde')
        ax_2.set_zlabel('Exact volatility: sigma_tilde')

        err_surf_sigma = torch.reshape(sigma_tilde_ref, [N,m]) - torch.reshape(inferred_sigma, [N,m])

        ax_3 = fig.add_subplot(1,3,3, projection='3d')
        ax_3.plot_surface(torch.reshape(k_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(t_tilde, [N,m]).detach().cpu().numpy(), torch.reshape(err_surf_sigma, [N,m]).detach().cpu().numpy(), cmap=cm.inferno,linewidth=0)
        ax_3.set_ylabel('Maturity: t_tilde')
        ax_3.set_xlabel('Strike price: k_tilde')
        ax_3.set_zlabel('Error sigma_tilde')

        plt.savefig(os.path.join(self.dirname,f'inferred_sigma_error_{step}.png'))
        if show:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1,2, figsize=[12,2], dpi=450)

        ax[0].semilogy(loss_phi_list, label='loss_phi')
        ax[0].semilogy(loss_dupire_list, label='loss_dupire')
        ax[0].semilogy(loss_reg_list, label='loss_reg')
        ax[0].semilogy(loss_boundary_list, label='loss_boundary')
        ax[0].semilogy(loss_arb_list, label='loss_arbitrage')
        ax[0].legend(loc='upper right')

        ax[1].plot(error_sigma_list, label='relative error sigma')
        ax[1].legend(loc='upper right')

        plt.savefig(os.path.join(self.dirname,f'losses_{step}.png'))
        if show:
            plt.show()
        plt.close()

    def save_model_and_results(self, step):
        option_model = self.NN_phi_tilde
        volatility_model = self.NN_eta_tilde
        option_optimizer = self.optimizer_NN_phi
        volatility_optimizer = self.optimizer_NN_eta

        option_model_path=os.path.join(self.dirname,f'comprehensive_option_model_{step}.pth')
        volatility_model_path=os.path.join(self.dirname,f'comprehensive_volatility_model_{step}.pth')

        option_model_info = {
            'model_state_dict': option_model.state_dict(),
            'optimizer_state_dict': option_optimizer.state_dict(),
            'model_architecture': option_model,
            }

        volatility_model_info = {
            'model_state_dict': volatility_model.state_dict(),
            'optimizer_state_dict': option_optimizer.state_dict(),
            'model_architecture': volatility_model,
            }

        torch.save(option_model_info, option_model_path)
        torch.save(volatility_model_info, volatility_model_path)

    def load_model_and_optimizer(self):
        option_model_path='comprehensive_option_model.pth'
        volatility_model_path='comprehensive_volatility_model.pth'
        def load_model(model_path, model, optimizer):
            model_info = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(model_info['model_state_dict'])
            optimizer.load_state_dict(model_info['optimizer_state_dict'])
            return model, optimizer
        option_model, option_optimizer = load_model(option_model_path, option_model, option_optimizer)
        volatility_model, volatility_optimizer = load_model(volatility_model_path, volatility_model, volatility_optimizer)
        return option_model, volatility_model

    def test(self):
        # Timing the computation of loss_phi
        time_0 = time.time()
        loss_phi = self.loss_phi_cal()
        time_1 = time.time()
        print(f'loss_phi = {loss_phi.item()}, computation time = {time_1 - time_0:.4f} seconds')

        # Timing the computation
        time_0 = time.time()
        loss_dupire, loss_reg, loss_arb = self.loss_dupire_cal()
        time_1 = time.time()

        print(f'loss_dupire = {loss_dupire.item()}, loss_reg = {loss_reg.item()}, loss_arb = {loss_arb.item()}, computation time = {time_1 - time_0:.4f} seconds')
        print()

        # Run the training step and time it
        time_0 = time.time()
        loss_phi, loss_dupire, loss_reg, loss_boundary, loss_arb = self.train_step()
        time_1 = time.time()

        print(f'computation time = {time_1 - time_0:.4f} seconds')
        print(f'loss_phi = {loss_phi}, loss_dupire = {loss_dupire}, loss_reg = {loss_reg}, loss_arb = {loss_arb}, loss_boundary = {loss_boundary}')
        print()

        # Run another training step for testing
        time_0 = time.time()
        loss_phi, loss_dupire, loss_reg, loss_boundary, loss_arb = self.train_step()
        time_1 = time.time()

        print(f'computation time = {time_1 - time_0:.4f} seconds')
        print(f'loss_phi = {loss_phi}, loss_dupire = {loss_dupire}, loss_reg = {loss_reg}, loss_arb = {loss_arb}, loss_boundary = {loss_boundary}')
        print()

        self.plot_res([],[],[],[],[],[], step=-1)

    def run(self):

        t_tilde = self.t_tilde
        k_tilde = self.k_tilde
        k_min = self.k_min
        k_max = self.k_max
        t_min = self.t_min
        t_max = self.t_max
        r = self.r

        # Lists to store the losses and errors
        loss_phi_list = []
        loss_dupire_list = []
        loss_reg_list = []
        loss_boundary_list = []
        loss_arb_list = []
        error_sigma_list = []

        # Training loop
        for iter_ in range(self.num_epochs+1):
            # Perform a training step
            loss_phi, loss_dupire, loss_reg, loss_boundary, loss_arb = self.train_step()

            # Append losses
            loss_phi_list.append(loss_phi)
            loss_dupire_list.append(loss_dupire)
            loss_reg_list.append(loss_reg)
            loss_boundary_list.append(loss_boundary)
            loss_arb_list.append(loss_arb)

            T_ = self.inverse_scaling_function(t_tilde, t_min, t_max)
            K_ = torch.exp(r * T_) * self.inverse_scaling_function(k_tilde, k_min, k_max)
            
            # Compute sigma_nn from the neural network
            sigma_nn = torch.sqrt(2 * self.neural_eta_tilde(t_tilde, k_tilde) / (t_max - t_min))
            
            # Compute exact sigma
            sigma_exact = self.exact_sigma_torch(T_, K_)
            
            # Compute error in sigma
            error_sigma = torch.mean(torch.abs(sigma_nn - sigma_exact) / sigma_exact)
            error_sigma_list.append(error_sigma.item())

            # Print progress every print_epochs iterations
            if iter_ % self.print_epochs == 0:
                print(f'iter = {iter_}: loss_phi = {loss_phi_list[-1]}, loss_dupire = {loss_dupire_list[-1]}, loss_arb = {loss_arb_list[-1]}, loss_reg = {loss_reg}, loss_boundary = {loss_boundary}, error_sigma = {error_sigma_list[-1]}')
                print(f"Iteration {iter_}: Learning rate for NN_phi: {self.optimizer_NN_phi.param_groups[0]['lr']}, Learning rate for NN_eta: {self.optimizer_NN_eta.param_groups[0]['lr']}")

            if iter_ % self.save_epochs == 0:
                self.plot_res(loss_phi_list, loss_dupire_list, loss_reg_list, loss_boundary_list, loss_arb_list, error_sigma_list, iter_)
                self.save_model_and_results(iter_)