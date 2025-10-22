import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn


class Hashgrid_MLP(nn.Module):
    def __init__(self, encoding_config, mlp_config):
        super().__init__()

        self.n_levels = encoding_config['n_levels']
        self.n_features_per_level = encoding_config['n_features_per_level']
        self.log2_hashmap_size = encoding_config['log2_hashmap_size']
        self.base_resolution = encoding_config['base_resolution']
        self.max_resolution = encoding_config['max_resolution']
        self.per_level_scale = self.get_per_level_scale()

        self.dim_hidden = mlp_config['dim_hidden']
        self.num_layers = mlp_config['num_layers']
        self.std = mlp_config['std']

        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale
        }

        self.mlp_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.dim_hidden,
            "n_hidden_layers": self.num_layers - 1
        }

        self.hash_mlp = tcnn.NetworkWithInputEncoding(2, 3, self.encoding_config, self.mlp_config)

        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.hash_mlp.parameters():
            if param.requires_grad:
                nn.init.normal_(param, mean=0.0, std=self.std)

    def get_per_level_scale(self):
        return (self.max_resolution / self.base_resolution) ** (1 / (self.n_levels - 1))

    def forward(self, x):
        return self.hash_mlp(x)