# import sys
import os

import torch
# from copy import deepcopy
import numpy as np

from zebanas.spaces.operations_v2 import OperationPoolV2
from tqdm import tqdm
import subprocess


# torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.set_num_threads(4)

NETWORKS_CHANNELS = [16, 24, 48, 80, 128, 192]
STRIDES = [1, 1, 2, 1, 2, 1]

INPUT_SIZE = 32
EXPAND_RATIOS = [3, 4, 6]

OPS_LOWER = [0]
OPS_UPPER = [11]

SAMPLES = []

TABLES = {}

CLOCK_SPEED = 1350
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")

device = torch.device("cpu")
repetitions = 10_000

model_list = []

for i in range(len(NETWORKS_CHANNELS[:-1])):
    in_chn = NETWORKS_CHANNELS[i]
    out_chn = NETWORKS_CHANNELS[i+1]
    stride = STRIDES[i+1]

    for expand_ratio in EXPAND_RATIOS:
        for op_idx in range(12):
            for j in range(2):
                if j > 0:
                    stride = 1
                    in_chn = out_chn
                model = OperationPoolV2(
                    expand_ratio,
                    in_chn,
                    out_chn,
                    stride,
                    op_idx
                )()
                input_shape = (1, in_chn, INPUT_SIZE, INPUT_SIZE)
                id = (i, j, expand_ratio, op_idx)
                model_list.append((id, model, input_shape))
                if stride > 1:
                    INPUT_SIZE = INPUT_SIZE // 2

print("Number of samples:", len(model_list))


def set_clock_speed():
    process = subprocess.Popen(
        "nvidia-smi",
        stdout=subprocess.PIPE,
        shell=True
    )
    stdout, _ = process.communicate()
    process = subprocess.run(
        f"sudo nvidia-smi -pm ENABLED -i {DEVICE}",
        shell=True
    )
    process = subprocess.run(
        f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}",
        shell=True
    )


def reset_clock_speed():
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)


def flush_cache(model, xs=None):
    if xs is not None:
        for x in xs:
            x.zero_()
    [p.data.zero_() for p in model.parameters()]


for id, model, input_shape in tqdm(model_list):
    set_clock_speed()
    model.to(device)
    x = torch.rand(input_shape).to(device)

    with torch.no_grad():
        # Warmup steps
        for _ in range(100):
            _ = model(x)

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(repetitions)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(repetitions)
        ]

        for i in range(repetitions):
            torch.cuda._sleep(1_000_000)
            start_events[i].record()
            _ = model(x)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg = np.mean(times)
        std = np.std(times)
        print(f"*** {avg} " + u"\u00B1" + f" {std} ***")

        TABLES[id] = {"mean": avg, "std": std}
    flush_cache(model, x)
    reset_clock_speed()

torch.save(TABLES, "zebanas/checkpoints/latency/latency_c10_cpu.pth")
