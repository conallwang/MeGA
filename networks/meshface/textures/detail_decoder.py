import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvUpBlock


class DynamicDecoder(nn.Module):
    def __init__(self, cfg):
        super(DynamicDecoder, self).__init__()

        self.neural = cfg.get("training.neural_texture", True)
        z_dim = cfg["flame.n_expr"] + cfg["flame.n_pose"]  # 400
        outdim = cfg["training.tex_ch"] if self.neural else 3
        self.z_fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, 64, base),
            ConvUpBlock(64, 32, base * 2),
            ConvUpBlock(32, 32, base * 4),
            ConvUpBlock(32, 16, base * 8),
            ConvUpBlock(16, 16, base * 16),
            ConvUpBlock(16, 8, base * 32),
            ConvUpBlock(8, outdim, base * 64),
        )

    def forward(self, z):
        B, _ = z.shape

        z_code = self.z_fc(z)
        z_code = z_code.view(B, 4, 8, 8)  # [B, 256] --> [B, 4, 8, 8]
        return self.upsample(z_code)


class ViewDecoder(nn.Module):
    def __init__(self, cfg):
        super(ViewDecoder, self).__init__()

        self.neural = cfg.get("training.neural_texture", True)
        tex_ch = cfg["training.tex_ch"] if self.neural else 3
        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(3, 64, base),
            ConvUpBlock(64, 32, base * 2),
            ConvUpBlock(32, 32, base * 4),
            ConvUpBlock(32, 16, base * 8),
            ConvUpBlock(16, 16, base * 16),
            ConvUpBlock(16, 8, base * 32),
            ConvUpBlock(8, tex_ch, base * 64),
        )

    def forward(self, view):
        return self.upsample(view)
