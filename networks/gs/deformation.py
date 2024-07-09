import numpy as np
import torch
import torch.nn as nn

from utils import positional_encoding


class DeformMLP(nn.Module):
    def __init__(self, cfg, attr_dims, num_freqs=6):
        super().__init__()

        self.num_freqs = cfg.get("gs.pe.num_freqs", num_freqs)
        self.include_input = cfg["training.pe.include_input"]
        self.log_sampling = cfg["training.pe.log_sampling"]
        self.input_ch = 3 * self.include_input + 3 * 2 * self.num_freqs
        # self.input_ch = 3

        n_expr = cfg["flame.n_expr"]

        outdim = 0
        for attr in cfg["gs.deform_attr"]:
            outdim += attr_dims[attr]

        # Build expr deform net
        deform_layers = cfg.get("gs.deform_layers", [128, 64])
        dims = [self.input_ch + n_expr] + deform_layers + [outdim]
        self.deform_expr = self._build_mlp(dims)

    def _build_mlp(self, dims):
        num_layers = len(dims)
        layers = []

        for l in range(0, num_layers - 1):
            cin, cout = dims[l], dims[l + 1]
            lin = nn.Linear(cin, cout)

            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(cout))

            if l == (num_layers - 2):
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight, 0.0)
            layers.append(lin)

            if l < (num_layers - 2):
                layers.append(nn.Softplus())

        return nn.Sequential(*layers)

    def forward(self, xyz, expr_params):
        xyz_pe = positional_encoding(xyz, self.num_freqs, self.include_input, self.log_sampling)

        input_tensor = torch.cat([xyz_pe, expr_params], dim=-1)
        offsets = self.deform_expr(input_tensor)

        return offsets
