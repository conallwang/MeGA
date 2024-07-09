import torch
import torch.nn as nn


class Conv2dBNUB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dBNUB, self).__init__()
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, feature_size[0], feature_size[1]))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x) + self.bias[None, ...])
        return out


# A Conv Upsample Block
class ConvUpBlock(nn.Module):
    def __init__(self, cin, cout, feature_size):
        super(ConvUpBlock, self).__init__()

        # self.conv1 = Conv2dWNUB(cin, cout, feature_size, 3, padding=1)
        self.conv1 = Conv2dBNUB(cin, cout, feature_size, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, 4 * cout, 1)

        self.relu = nn.LeakyReLU(0.2)

        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.relu(self.conv2(self.conv1(x))))
