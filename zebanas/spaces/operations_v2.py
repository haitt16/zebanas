import torch
import torch.nn as nn
# from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from .layers import (
    LayerNorm,
    DepthwiseConv,
    GhostModule
)

from .operations import (
    ConvolutionOperation,
    DepthwiseOperation,
    SqueezeExcitationOperation
)

from .utils import _adjust_channels


class DepthwiseOperationV2(nn.Module):
    def __init__(self, expand_ratio, in_chn, out_chn, kernel_size, stride):
        super().__init__()

        mid_chn = _adjust_channels(in_chn, expand_ratio)

        self.conv1 = DepthwiseOperation(in_chn, mid_chn, kernel_size, stride)
        self.conv2 = SqueezeExcitationOperation(
            mid_chn,
            out_chn,
            1,
            expand_ratio
        )

        self.use_res = in_chn == out_chn and stride == 1

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)

        if self.use_res:
            o = x + o

        return o


class ConvolutionOperationV2(nn.Module):
    def __init__(self, expand_ratio, in_chn, out_chn, kernel_size, stride):
        super().__init__()
        mid_chn = _adjust_channels(in_chn, expand_ratio)

        self.conv1 = ConvolutionOperation(in_chn, mid_chn, kernel_size, stride)
        self.conv2 = SqueezeExcitationOperation(
            mid_chn,
            out_chn,
            1,
            expand_ratio
        )
        self.use_res = in_chn == out_chn and stride == 1

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)

        if self.use_res:
            o = o + x

        return o


class GhostOperationV2(nn.Module):
    def __init__(self, expand_ratio, in_chn, out_chn, kernel_size, stride):
        super().__init__()
        mid_chn = _adjust_channels(in_chn, expand_ratio)
        self.stride = stride

        self.conv1 = GhostModule(in_chn, mid_chn, relu=True)

        if stride > 1:
            self.dw = DepthwiseConv(
                mid_chn, mid_chn,
                kernel_size, stride,
                activation=None
            )
        self.conv2 = GhostModule(mid_chn, out_chn, relu=False)

        self.use_shortcut = in_chn != out_chn or stride != 1

        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chn, in_chn,
                    kernel_size, stride,
                    (kernel_size-1)//2, groups=in_chn,
                    bias=False
                ),
                nn.BatchNorm2d(in_chn),
                nn.Conv2d(
                    in_chn, out_chn, 1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(out_chn)
            )

    def forward(self, x):
        o = self.conv1(x)
        if self.stride > 1:
            o = self.dw(o)
        o = self.conv2(o)

        if self.use_shortcut:
            x = self.shortcut(x)

        o = o + x
        return o


class ConvNeXtOperation(nn.Module):
    def __init__(
        self,
        expand_ratio,
        in_chn,
        out_chn,
        kernel_size,
        stride
    ):
        super().__init__()

        self.mode = None
        if in_chn != out_chn:
            if stride > 1:
                self.downsample = nn.Sequential(
                    LayerNorm(in_chn, data_format="channels_first"),
                    nn.Conv2d(in_chn, out_chn, 2, stride)
                )
                self.mode = "0"
            else:
                self.conv = nn.Sequential(
                    LayerNorm(in_chn, data_format="channels_first"),
                    nn.Conv2d(in_chn, out_chn, 3, stride, 1)
                )
                self.mode = "1"
        else:
            self.mode = "2"
            mid_chn = _adjust_channels(in_chn, expand_ratio)
            self.dwconv = nn.Conv2d(
                in_chn, in_chn,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size-1)//2,
                groups=in_chn
            )
            self.norm = LayerNorm(in_chn)
            self.pwconv1 = nn.Linear(in_chn, mid_chn)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(mid_chn, out_chn)
            layer_scale_init_value = 1e-6
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(out_chn),
                requires_grad=True
            )

    def forward(self, x):
        if self.mode == "0":
            return self.downsample(x)
        elif self.mode == "1":
            return self.conv(x)
        else:
            input = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)

            x = input + x
            return x


OPERATIONS_CLASSES = [
    # op_class, k
    (GhostOperationV2, 3),
    (GhostOperationV2, 5),
    (GhostOperationV2, 7),
    (DepthwiseOperationV2, 3),
    (DepthwiseOperationV2, 5),
    (DepthwiseOperationV2, 7),
    (ConvolutionOperationV2, 3),
    (ConvolutionOperationV2, 5),
    (ConvolutionOperationV2, 7),
    (ConvNeXtOperation, 3),
    (ConvNeXtOperation, 5),
    (ConvNeXtOperation, 7),
]


class OperationPoolV2:
    def __init__(
        self,
        expand_ratio,
        in_chn, out_chn,
        stride, index
    ):
        op_class, k = OPERATIONS_CLASSES[index]

        self.op = op_class(
            expand_ratio,
            in_chn, out_chn,
            k, stride
        )

    def __call__(self):
        return self.op
