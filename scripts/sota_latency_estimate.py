# import sys
import os

import torch
# from copy import deepcopy
import numpy as np
from tqdm import tqdm
import subprocess

from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from proxylessnas.proxyless_nas.model_zoo import proxyless_base
from scripts.sotas.convnext import convnext_tiny
from scripts.sotas.ghostnet import ghostnet
from zebanas.evaluators.zico import ZicoProxyV2
from zebanas.data.vision.cifar10 import DataLoaderforSearchGetter
# from zebanas.genetic.chromosome import ChromosomeV2
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create
api = create("/home/haitt/workspaces/codes/nas-bench/NATS-Bench/api/NATS-tss-v1_0-3ffb9-simple", "tss", fast_mode=True)
config = api.get_net_config(15624, 'cifar10')
info = api.get_more_info(15624, "cifar10", hp="200")
print(info)
# torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.set_num_threads(1)

CLOCK_SPEED = 1350
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")

device = torch.device("cpu")
repetitions = 1_000
TABLES = {}

model_list = [
    # ("efficientnet_b0", efficientnet_b0(num_classes=10), 224),
    # ("efficientnet_b1", efficientnet_b1(num_classes=10), 240),
    # ("efficientnet_b2", efficientnet_b2(num_classes=10), 288),
    # ("efficientnet_b3", efficientnet_b3(num_classes=10), 300),
    # ("resnet18", resnet18(num_classes=10), 224),
    # ("resnet34", resnet34(num_classes=10), 224),
    ("proxyless_cifar", proxyless_base(pretrained=False, net_config="https://raw.githubusercontent.com/han-cai/files/master/proxylessnas/proxyless_cifar.config"), 32),
    ("best_nb", get_cell_based_tiny_net(config), 32)
    # ("resnet50", resnet50(num_classes=10), 224),
    # ("resnet101", resnet101(num_classes=10), 224),
    # ("mobilenet_v2", mobilenet_v2(num_classes=10), 224),
    # ("mobilenet_v3_small", mobilenet_v3_small(num_classes=10), 224),
    # ("mobilenet_v3_large", mobilenet_v3_large(num_classes=10), 224),
    # ("convnext_tiny", convnext_tiny(num_classes=10), 224),
    # ("ghostnet", ghostnet(num_classes=10), 224)
]

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


def flush_cache(model, xs=None, no_grad=True):
    if xs is not None:
        for x in xs:
            x.zero_()
    [p.data.zero_() for p in model.parameters()]

    if not no_grad:
        [p.grad.zero_() for p in model.parameters()]


for id, model, input_shape in tqdm(model_list):
    set_clock_speed()
    model.to(device)
    x = torch.rand(1, 3, input_shape, input_shape).to(device)

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

        for i in tqdm(range(repetitions)):
            torch.cuda._sleep(1_000_000)
            start_events[i].record()
            _ = model(x)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg = np.mean(times)
        std = np.std(times)
        print(f"*** {avg} " + u"\u00B1" + f" {std} ***")

        # TABLES[id] = {"latency": {"mean": avg, "std": std}}
    flush_cache(model, [x])
    reset_clock_speed()

# TABLES = torch.load("zebanas/checkpoints/latency/sota_c10_gpu.pth")
evaluator = ZicoProxyV2(torch.nn.CrossEntropyLoss(), 30)

for id, model, input_shape in model_list:
    set_clock_speed()
    dataloader = DataLoaderforSearchGetter(
        "/home/haitt/workspaces/data/vision/cifar10",
        2, input_shape, input_shape, 2
    )
    dataloader = dataloader.load()

    score_list = []

    for _ in tqdm(range(30)):
        torch.cuda._sleep(1_000_000)
        score = evaluator.get_zico(model, dataloader, "cuda")
        score_list.append(score)
    torch.cuda.synchronize()
    avg = np.mean(score_list)
    std = np.std(score_list)
    print(f"*** {avg} " + u"\u00B1" + f" {std} ***")

    # TABLES[id]["score"] = {"mean": avg, "std": std}
    print(id, avg, std)
    reset_clock_speed()

print(TABLES)
# torch.save(TABLES, "zebanas/checkpoints/latency/sota_c10_gpu.pth")
