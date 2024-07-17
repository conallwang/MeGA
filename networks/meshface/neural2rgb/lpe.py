import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_sample1d(input, grid, mode="bilinear", padding_mode="zeros", align_corners=True):
    """Grid Sample1d

    Args:
        input (BxCxLin): a tensor to sample from
        grid (BxHoutxWout): grid tensor
        mode (str, optional): Defaults to 'bilinear'
        padding_mode (str, optional): Defaults to 'zeros'.
        align_corners (bool, optional): Defaults to True.

    Returns:
        z (BxCxHoutxWout)
    """
    input = input.unsqueeze(-1)  # [B, C, Lin, 1]

    grid = torch.stack([-torch.ones_like(grid).to(grid.device), grid], dim=-1)  # [B, 1, N, 2]
    z = F.grid_sample(
        input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )  # [B, C, Hout, Wout]
    return z


class PE(nn.Module):
    def __init__(self, cfg):
        super(PE, self).__init__()

        self.input_dims = cfg["input_dims"]
        self.num_freqs = cfg["num_freqs"]
        self.max_freq_log2 = self.num_freqs - 1
        self.periodic_fns = cfg["periodic_fns"]
        self.log_sampling = cfg["log_sampling"]
        self.include_input = cfg["include_input"]
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class LPE(nn.Module):
    def __init__(self, cfg):
        super(LPE, self).__init__()
        self.m_u = nn.Parameter(torch.FloatTensor(2, 10000).uniform_(-1, 1), requires_grad=True)
        self.m_v = nn.Parameter(torch.FloatTensor(2, 10000).uniform_(-1, 1), requires_grad=True)
        self.out_dim = 4

    def forward(self, uv, vis=False):
        """
        Args:
            uv (float32, -1~1): [N, 2]
        """
        u = uv[:, 0]
        v = uv[:, 1]

        m_u = grid_sample1d(self.m_u[None], u[None, None])[0, :, 0]  # [2, N]
        m_v = grid_sample1d(self.m_v[None], v[None, None])[0, :, 0]
        uv_pe = torch.cat([m_u, m_v], dim=0).permute((1, 0))  # [N, 4]

        return uv_pe
