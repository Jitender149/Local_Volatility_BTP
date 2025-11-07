from imports import *

class NetPhiTilde(nn.Module):
    def __init__(self, nn_params):
        super(NetPhiTilde, self).__init__()
        [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate] = nn_params
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.layers(x))

class NetEtaTilde(nn.Module):
    def __init__(self, nn_params):
        super(NetEtaTilde, self).__init__()
        [input_size, output_size, hidden_layers, num_neurons, neurons_per_layer, activation, use_batch_norm, dropout_rate] = nn_params
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.layers(x))