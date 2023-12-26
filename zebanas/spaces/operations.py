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
    def __init__(self, in_chn, out_chn, kernel_size, stride, relu):
        super().__init__()

        layers = [GhostModule(in_chn, out_chn, relu=relu)]

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
    def __init__(self, in_chn, out_chn, stride, expand_ratio):
        super().__init__()
        layers = []

        squeeze_chn = max(1, in_chn // 4 // expand_ratio)

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


OPERATIONS_CLASSES = [
    # op_class, k
    (NoneOperation, None),
    (IdentityOperation, None),
    (SqueezeExcitationOperation, None),
    (GhostOperation, 3),
    (GhostOperation, 5),
    (GhostOperation, 7),
    (DepthwiseOperation, 3),
    (DepthwiseOperation, 5),
    (DepthwiseOperation, 7),
    (ConvolutionOperation, 3),
    (ConvolutionOperation, 5),
    (ConvolutionOperation, 7),
]


class OperationPool:
    def __init__(
        self,
        in_chn, out_chn, stride,
        index, expand_ratio=0, relu=True
    ):
        op_class, k = OPERATIONS_CLASSES[index]

        if index in [3, 4, 5]:
            self.op = op_class(in_chn, out_chn, k, stride, relu=relu)
        elif index >= 6:
            self.op = op_class(in_chn, out_chn, k, stride)
        elif index == 0:
            self.op = op_class(out_chn, stride)
        elif index == 1:
            self.op = op_class(in_chn, out_chn, stride)
        elif index == 2:
            self.op = op_class(in_chn, out_chn, stride, expand_ratio)

    def __call__(self):
        return self.op
