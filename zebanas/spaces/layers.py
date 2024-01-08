# import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Conv2dNormActivation


class PointwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation=nn.ReLU
    ):
        super().__init__()

        self.pw = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            norm_layer=nn.BatchNorm2d,
            activation_layer=activation,
            bias=False
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
        activation=nn.ReLU
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
            activation_layer=activation,
            bias=False
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
        init_channels = oup // ratio
        new_channels = init_channels*(ratio-1)

        self.primary_conv = Conv2dNormActivation(
            in_channels=inp,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            bias=False
        )
        self.cheap_operation = Conv2dNormActivation(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=1,
            padding=dw_size//2,
            groups=init_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            bias=False
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_last"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape,
                self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
