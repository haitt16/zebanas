import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from functools import partial
import math

from .layers import (
    Identity,
    PointwiseConv,
    DepthwiseConv,
    GhostModule
)


class ConvolutionOperation(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride):
        super().__init__()

        self.conv = Conv2dNormActivation(
            in_channels=in_chn,
            out_channels=out_chn,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size-1)//2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseOperation(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride):
        super().__init__()

        layers = []

        if in_chn != out_chn:
            layers.append(
                PointwiseConv(in_chn, out_chn)
            )

        layers.append(
            DepthwiseConv(out_chn, out_chn, kernel_size, stride)
        )

        self.dw = nn.Sequential(*layers)

    def forward(self, x):
        return self.dw(x)


class GhostOperation(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride):
        super().__init__()

        layers = [GhostModule(in_chn, out_chn, relu=True)]

        if stride > 1:
            layers.append(
                DepthwiseConv(
                    out_chn, out_chn,
                    kernel_size, stride,
                    activation=None
                )
            )

        self.ghost = nn.Sequential(*layers)

    def forward(self, x):
        return self.ghost(x)


class SqueezeExcitationOperation(nn.Module):
    def __init__(self, in_chn, out_chn, stride):
        super().__init__()

        layers = []
        squeeze_chn = max(1, in_chn // 4)

        layers.append(
            SqueezeExcitation(
                in_chn,
                squeeze_chn,
                partial(nn.SiLU, inplace=True)
            )
        )
        if in_chn != out_chn or stride > 1:
            layers.append(
                PointwiseConv(in_chn, out_chn, stride, None)
            )

        self.se = nn.Sequential(*layers)

    def forward(self, x):
        return self.se(x)


class IdentityOperation(nn.Module):
    def __init__(self, in_chn, out_chn, stride):
        super().__init__()

        layers = []
        if in_chn != out_chn or stride > 1:
            layers.append(
                PointwiseConv(in_chn, out_chn, stride, None)
            )
        layers.append(Identity())

        self.identity = nn.Sequential(*layers)

    def forward(self, x):
        return self.identity(x)


class NoneOperation(nn.Module):
    def __init__(self, out_chn, stride):
        super().__init__()
        self.out_chn = out_chn
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.size()
        nh = math.floor(((h-1)/self.stride) + 1)
        nw = math.floor(((w-1)/self.stride) + 1)
        return torch.zeros(b, self.out_chn, nh, nw).to(x.device)

# class NoneOperation(nn.Module):

#     def __init__(self, out_chn, stride):
#         super(NoneOperation, self).__init__()
#         self.stride = stride

#     def forward(self, x):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)


class OperationPool:
    def __init__(self, in_chn, out_chn, stride, index):
        pool = [
            NoneOperation(out_chn, stride),
            IdentityOperation(in_chn, out_chn, stride),
            SqueezeExcitationOperation(in_chn, out_chn, stride),
            # GhostOperation(in_chn, out_chn, 3, stride),
            # GhostOperation(in_chn, out_chn, 5, stride),
            # GhostOperation(in_chn, out_chn, 7, stride),
            DepthwiseOperation(in_chn, out_chn, 3, stride),
            DepthwiseOperation(in_chn, out_chn, 5, stride),
            DepthwiseOperation(in_chn, out_chn, 7, stride),
            ConvolutionOperation(in_chn, out_chn, 3, stride),
            ConvolutionOperation(in_chn, out_chn, 5, stride),
            ConvolutionOperation(in_chn, out_chn, 7, stride),
        ]

        self.op = pool[index]

    def __call__(self):
        return self.op
