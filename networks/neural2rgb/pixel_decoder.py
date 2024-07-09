import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from collections import OrderedDict


class SineLayer(nn.Module):
    # refer to https://github1s.com/vsitzmann/siren

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0
                )

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output
        # return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations["_".join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class PixelDecoder(nn.Module):
    def __init__(self, cin=16, pe_dim=8, first_omega_0=30, hidden_omega_0=30.0, xyz_cond=True):
        super().__init__()

        self.pe_dim = pe_dim
        self.xyz_cond = xyz_cond

        # encode xyz to 4-dim vector
        self.xyz_enc = Siren(
            in_features=3,
            hidden_features=4,
            hidden_layers=0,
            out_features=4,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        if not xyz_cond:
            cin -= 4
        self.final_siren = nn.Sequential(
            Siren(
                in_features=cin,
                hidden_features=8,
                hidden_layers=2,
                out_features=3,
                outermost_linear=True,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, latents):
        if self.xyz_cond:
            xyz, uv_pe, z = latents[:, :3], latents[:, 3 : (3 + self.pe_dim)], latents[:, (3 + self.pe_dim) :]
            x = self.xyz_enc(xyz)  # Bx4
            inputs = [x, uv_pe, z]
        else:
            uv_pe, z = latents[:, : self.pe_dim], latents[:, self.pe_dim :]
            inputs = [uv_pe, z]

        enc_inputs = torch.cat(inputs, dim=-1)  # Bx16
        colors = self.final_siren(enc_inputs)  # Bx3

        return colors


# if __name__ == "__main__":
#     model = PixelDecoder()

#     print(count_parameters(model))
