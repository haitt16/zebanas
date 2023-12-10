import sys
import os

import torch
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zebanas.spaces.modules import Block
from tqdm import tqdm

CHANNELS = ["32-16", "16-16", "16-24", "24-24", "24-40", "40-40", "40-80", "80-80", "80-112", "112-112", "112-192", "192-192", "192-360", "360-360"]
RESOLUTIONS = ["112-112", "112-112", "112-56", "56-56", "56-28", "28-28", "28-14", "14-14", "14-14", "14-14", "14-7", "7-7", "7-7", "7-7"]
EXPANDS_RATIO = [1, 2, 4, 6]

OPS_LOWER = [3, 1, 0, 0, 0]
OPS_UPPER = [11, 11, 0, 1, 0]

SAMPLES = []

TABLES = {}


def generate(seq, i):
    if i == len(seq):
        SAMPLES.append(deepcopy(seq))
        return
    for val in range(OPS_LOWER[i], OPS_UPPER[i]+1):
        seq[i] = val
        generate(seq, i+1)


generate(deepcopy(OPS_LOWER), 0)
device = torch.device("cuda")
repetitions = 300

for index, (channels, resolutions) in enumerate(zip(CHANNELS, RESOLUTIONS)):
    in_chn, out_chn = channels.split("-")
    in_chn, out_chn = int(in_chn), int(out_chn)

    in_res, out_res = resolutions.split("-")
    in_res, out_res = int(in_res), int(out_res)
    if in_res != out_res:
        stride = 2
    else:
        stride = 1

    for expands_ratio in EXPANDS_RATIO:
        for ops in tqdm(SAMPLES):
            timings = []

            if index % 2 == 0:
                cell_idx = index // 2
                btype = 0
            else:
                cell_idx = (index - 1) // 2
                btype = 1
            code_id = [cell_idx, btype, expands_ratio, ops[0], ops[1], ops[3]]
            block = Block(
                expands_ratio,
                in_chn,
                out_chn,
                stride,
                ops
            )
            block.to(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            dummy_input = torch.rand(1, in_chn, in_res, in_res).to(device)
            # warm up
            for _ in range(10):
                _ = block(dummy_input)

            with torch.no_grad():
                for rep in range(repetitions):
                    starter.record()
                    _ = block(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings.append(curr_time)

                mean_latency = sum(timings) / repetitions

            TABLES[tuple(code_id)] = mean_latency

torch.save(TABLES, "latency_table.pth")