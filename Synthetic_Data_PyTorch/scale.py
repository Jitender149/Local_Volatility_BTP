from imports import *

class ScaleQuantities:
    def __init__(self, S_0, r, t_all, S_matrix, N, m, strikes, times, data_type, show, device, dirname):
        self.S_0 = S_0
        self.r = r
        self.t_all = t_all
        self.N = N
        self.m = m
        self.strikes = strikes
        self.times = times
        self.S_matrix = S_matrix
        self.data_type = data_type
        self.show = show 
        self.device = device
        self.dirname = dirname

    def exact_phi(self, T, K):
        E_ = torch.cat([torch.reshape(torch.mean(torch.relu(torch.unsqueeze(self.S_T[i], dim=0) - torch.unsqueeze(K[i], dim=1)), dim=1), (1,-1)) for i in range(self.N)], dim=0)
        phi_ = torch.exp(-self.r * T) * E_
        return phi_

    def plot_options_maturities(self):
        fig = plt.figure(figsize=[14,6], dpi = 450)
        ax = fig.add_subplot(121)
        ax.scatter(self.t_tilde.cpu().numpy(), self.k_tilde.cpu().numpy(), s=0.1)
        ax.set_xlabel('t_tilde')
        ax.set_ylabel('k_tilde')

        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(torch.reshape(self.k_tilde, [self.N,self.m]).cpu().numpy(), torch.reshape(self.t_tilde, [self.N,self.m]).cpu().numpy(), torch.reshape(self.phi_tilde_ref, [self.N,self.m]).cpu().numpy())
        ax.set_ylabel('Scaled maturity: t_tilde')
        ax.set_xlabel('Scaled strike price: k_tilde')
        ax.set_zlabel('Scaled option price: phi_tilde')
        if self.show:
            plt.show()
        plt.savefig(os.path.join(self.dirname,'scaled_phi_t_k.png'))
        plt.close()

    def scaling_function(self, param, param_min, param_max):
        return (param - param_min) / (param_max - param_min)

    def inverse_scaling_function(self, param_scaled, param_min, param_max, square=False):
        if square:
            return (param_min + param_scaled*(param_max - param_min))/(param_max - param_min)
        else:
            return param_min + param_scaled*(param_max - param_min) 

    def run(self):

        # self.T = torch.repeat_interleave(torch.reshape(self.t_all[50:], (-1,1)), self.m, dim=1)
        self.T = torch.repeat_interleave(torch.tensor(np.reshape(self.times, (-1, 1)), dtype=self.data_type).to(self.device), self.m, dim=1)
        self.K = torch.tensor(np.repeat(np.reshape(self.strikes, (1, -1)), len(self.T), axis=0), dtype=self.data_type).to(self.device)
        self.S_T = self.S_matrix[50:]
        
        phi_exact = self.exact_phi(self.T, self.K)

        T_nn = torch.reshape(self.T, [-1,1])
        K_nn = torch.reshape(self.K, [-1,1])
        phi_ref = torch.reshape(phi_exact, [-1,1])

        t_nn = T_nn
        k_nn = torch.exp(-self.r*T_nn) * K_nn

        self.t_min = torch.min(t_nn)
        self.t_max = torch.max(t_nn)
        self.k_min = torch.min(k_nn)
        self.k_max = torch.max(k_nn)

        self.t_tilde = self.scaling_function(t_nn, self.t_min, self.t_max)
        self.k_tilde = self.scaling_function(k_nn, self.k_min, self.k_max)

        self.phi_tilde_ref = phi_ref / self.S_0

        self.t_tilde, self.k_tilde, self.phi_tilde_ref = self.t_tilde.to(self.device), self.k_tilde.to(self.device), self.phi_tilde_ref.to(self.device)

        self.plot_options_maturities()