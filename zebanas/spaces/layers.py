import math

import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation


class PointwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation=nn.SiLU
    ):
        super().__init__()

        self.pw = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            norm_layer=nn.BatchNorm2d,
            activation_layer=activation
        )

    def forward(self, x):
        return self.pw(x)


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation=nn.SiLU
    ):
        super().__init__()
        self.dw = Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size-1)//2,
                groups=in_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation
            )

    def forward(self, x):
        return self.dw(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GhostModule(nn.Module):
    def __init__(self,
                 inp, oup,
                 kernel_size=1,
                 ratio=2, dw_size=3,
                 stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels,
                kernel_size, stride, kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                3, 1, dw_size//2, groups=init_channels,
                bias=False
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
