import torch
import torch.nn as nn

from .layers import ConvUpBlock


class DispDecoder(nn.Module):
    def __init__(self, cfg, eyes=True):
        super(DispDecoder, self).__init__()
        self.eyes = eyes

        z_dim = cfg["flame.n_expr"] + cfg["flame.n_pose"]
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
            ConvUpBlock(16, 3, base * 16),  # add a block
        )

        self.lefteye_scale = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.lefteye_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3),  # only T, rotation is handled by eye poses.
        )
        torch.nn.init.constant_(self.lefteye_fc[-1].weight, 0.0)
        torch.nn.init.constant_(self.lefteye_fc[-1].bias, 0.0)
        self.lefteye_fc[-1].bias.data[0] = 1.0

        self.righteye_scale = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.righteye_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3),  # T
        )
        torch.nn.init.constant_(self.righteye_fc[-1].weight, 0.0)
        torch.nn.init.constant_(self.righteye_fc[-1].bias, 0.0)
        self.righteye_fc[-1].bias.data[0] = 1.0

    def forward(self, z):
        B, _ = z.shape

        z_code = self.z_fc(z)

        l_transform, r_transform = None, None
        if self.eyes:
            # eyes transform
            lT = self.lefteye_fc(z_code)
            rT = self.righteye_fc(z_code)

            ls = torch.exp(self.lefteye_scale[None].expand(B, -1))
            rs = torch.exp(self.righteye_scale[None].expand(B, -1))

            l_transform = torch.cat([ls, lT], dim=-1)
            r_transform = torch.cat([rs, rT], dim=-1)

        # decode disp map
        z_code = z_code.view(B, 4, 8, 8)  # [B, 256] --> [B, 4, 8, 8]
        return self.upsample(z_code), l_transform, r_transform
