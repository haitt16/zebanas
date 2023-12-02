from typing import Optional

import math


def _make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None
) -> int:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _adjust_channels(
    channels: int,
    width_mult: float,
    min_value: Optional[int] = None
) -> int:
    return _make_divisible(channels * width_mult, 8, min_value)


def adjust_depth(
    num_layers: int,
    depth_mult: float
):
    return int(math.ceil(num_layers * depth_mult))
