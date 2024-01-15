import math

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.ops.misc import Conv2dNormActivation

from .modules import Cell, Gecco2024Cell


class Network(nn.Module):
    def __init__(
        self,
        chromos,
        network_channels,
        strides,
        dropout,
        num_classes,
        last_channels,
        width_mult=1.0,
        depth_mult=1.0
    ):
        super().__init__()
        self.stem = Conv2dNormActivation(
            3, network_channels[0],
            kernel_size=3,
            stride=strides[0],
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU
        )

        cells = []

        for i in range(len(network_channels[:-1])):
            chromo = chromos[i]
            in_channels = network_channels[i]
            out_channels = network_channels[i+1]
            stride = strides[i+1]
            cells.append(
                Cell(
                    chromo, in_channels, out_channels,
                    stride, width_mult, depth_mult
                )
            )
        self.features = nn.Sequential(*cells)
        self.conv = Conv2dNormActivation(
                network_channels[-1],
                last_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU,
            )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class Gecco2024Network(nn.Module):
    def __init__(
        self,
        chromos,
        network_channels,
        strides,
        dropout,
        num_classes,
        last_channels,
    ):
        super().__init__()
        self.stem = Conv2dNormActivation(
            3, network_channels[0],
            kernel_size=3,
            stride=strides[0],
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU
        )

        cells = []
        total_blocks = sum(c[1] for c in chromos)
        start_block_id = 0
        for i in range(len(network_channels[:-1])):
            chromo = chromos[i]
            in_channels = network_channels[i]
            out_channels = network_channels[i+1]
            stride = strides[i+1]
            sd_probs = [0.2*(start_block_id + i)/total_blocks for i in range(chromo[1])]
            cells.append(
                Gecco2024Cell(
                    chromo,
                    in_channels,
                    out_channels,
                    stride,
                    sd_probs
                )
            )
            start_block_id += chromo[1]
        self.features = nn.Sequential(*cells)
        self.conv = Conv2dNormActivation(
                network_channels[-1],
                last_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU,
            )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
