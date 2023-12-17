import sys
import os

import numpy as np

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zebanas.spaces.modules import Block

o = open("latency.txt", "a")

code_id = sys.argv[1]
line_idx = sys.argv[2]

if int(line_idx) % 2 == 0:
    print(line_idx)

code_id = list(map(int, code_id.split()))

device = torch.device("cuda")
repetitions = 1000

block = Block(
    code_id[2],
    code_id[-3],
    code_id[-2],
    code_id[-1],
    code_id[3:6],
)
block.to(device)
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
dummy_input = torch.rand(1, code_id[-3], code_id[-4], code_id[-4]).to(device)
# warm up
for _ in range(100):
    _ = block(dummy_input)

timings = []

with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = block(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings.append(curr_time)

    mean_latency = sum(timings) / repetitions
    std_latency = np.std(timings)

# if code_id[:-4] == [0, 0, 4, 6, 7, 1]:
#     print(mean_latency)

o.write(f"{mean_latency}\t{std_latency}\n")
# print(code_id)
