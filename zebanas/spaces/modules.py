import torch.nn as nn

from .operations import OperationPool
from .utils import _adjust_channels, adjust_depth


class Connection(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if args[-1] == 0:
            return None
        self.edge = OperationPool(*args)()

    def forward(self, x):
        return self.edge(x)


class Block(nn.Module):
    def __init__(
        self,
        expand_ratio: int,
        in_chn: int,
        out_chn: int,
        stride: int,
        op_ids
    ):
        super().__init__()

        mid_chn = _adjust_channels(in_chn, expand_ratio)
        self.op_ids = op_ids

        self.use_res = in_chn == out_chn and stride == 1

        self.block1 = Connection(in_chn, mid_chn, stride, op_ids[0])
        if not self.use_res:
            block2 = None
        else:
            block2 = Connection(in_chn, out_chn, stride, op_ids[3])

        self.block2 = block2
        self.block3 = Connection(mid_chn, out_chn, 1, op_ids[1], expand_ratio)

    def forward(self, x):
        z = self.block1(x)

        if self.op_ids[3] != 0 and self.use_res:
            o = self.block2(x) + self.block3(z)
        else:
            o = self.block3(z)
        return o


class Cell(nn.Module):
    def __init__(
        self,
        chromo,
        in_channels,
        out_channels,
        stride,
        width_mult,
        depth_mult,
    ):
        super().__init__()

        nlayers = adjust_depth(chromo.nlayers, depth_mult)
        if nlayers > chromo.nlayers:
            for _ in range(nlayers - chromo.nlayers):
                chromo.expands.append(chromo.expands[-1])

        in_channels = _adjust_channels(in_channels, width_mult)
        out_channels = _adjust_channels(out_channels, width_mult)
        blocks = []

        for i in range(0, nlayers):
            if i > 0:
                in_channels = out_channels
                stride = 1
            blocks.append(
                Block(
                    chromo.expands[i],
                    in_channels,
                    out_channels,
                    stride,
                    chromo.operations
                )
            )

        self.cell = nn.Sequential(*blocks)

    def forward(self, x):
        return self.cell(x)
