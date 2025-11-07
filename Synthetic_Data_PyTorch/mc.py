from imports import *

class MonteCarloLocalVolatility:
    def __init__(self, S_0, r, x_, t_, d, M, N_t, dt, data_type, show, device, dirname):
        self.S_0 = S_0
        self.r = r
        self.x_ = x_
        self.t_ = t_
        self.d = d
        self.M = M
        self.N_t = N_t
        self.dt = dt
        self.data_type = data_type
        self.show = show
        self.device = device
        self.dirname = dirname

    def exact_sigma_np(self, t, x):
        y_ = np.sqrt(x + 0.1) * (t + 0.1)
        sigma_ = 0.2 + y_ * np.exp(-y_)
        return sigma_

    def exact_sigma_torch(self, t, x):
        y_ = torch.sqrt(x + 0.1) * (t + 0.1)
        sigma_ = 0.2 + y_ * torch.exp(-y_)
        return sigma_

    def plot_volatility(self):
        t_mesh, x_mesh = np.meshgrid(self.t_, self.x_)
        sigma_surf = np.array(self.exact_sigma_np(np.ravel(t_mesh), np.ravel(x_mesh)))
        sigma_surf_mesh = sigma_surf.reshape(x_mesh.shape)
        print('Exact local volatility surface')
        fig = plt.figure(figsize=[8,6], dpi = 450)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(t_mesh, x_mesh, sigma_surf_mesh)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('local volatility')
        if self.show:
            plt.show()
        plt.savefig(os.path.join(self.dirname,'local_volatility.png'))
        plt.close()

    def plot_price_trajectories(self):
        fig, ax = plt.subplots(figsize=[12, 3], dpi=450)
        plt.plot(self.t_all.cpu().numpy(), self.S_matrix[:,:1024].cpu().numpy(), lw=0.1)
        if self.show:
            plt.show()
        plt.savefig(os.path.join(self.dirname,'price_trajectories.png'))
        plt.close()

    def perform_monte_carlo(self):
        t_all = torch.reshape(torch.tensor(np.linspace(0, self.N_t*self.dt, self.N_t), dtype=self.data_type).to(self.device), (-1,1)) # exclude 0
        S_list = [torch.reshape(torch.tensor(np.full(self.M, self.S_0[0].item()), dtype=self.data_type).to(self.device), (1,self.M))]
        dW_list = torch.cat([torch.tensor(np.random.normal(0, 1, size=[self.N_t,1]) * np.sqrt(self.dt), dtype=self.data_type).to(self.device) for i in range(self.M)], dim=1)

        for i in range(self.N_t - 1):
            t_now = t_all[i]
            S_now = S_list[-1]
            S_new = S_now + self.r * S_now * self.dt + self.exact_sigma_torch(t_now, S_now) * S_now * dW_list[i]
            S_list.append(S_new)
        return S_list, t_all

    def run(self):
        self.plot_volatility()
        S_list, self.t_all = self.perform_monte_carlo()
        self.S_matrix = torch.cat(S_list, dim=0)
        self.plot_price_trajectories()
        print(f'S_t obtained by solving local volatility SDE M = {self.M} times from t = [0, {self.N_t*self.dt}]')

        return self.S_matrix, self.t_all