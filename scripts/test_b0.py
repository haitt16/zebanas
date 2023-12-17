import torch
from torchvision.models import efficientnet_b0, mobilenet_v2
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zebanas.evaluators.zico import ZicoProxy
from zebanas.spaces.modules import Block
from zebanas.spaces.model import Network


cfg = OmegaConf.load("/home/haitt/workspaces/codes/nas/zebanas/zebanas/configs/search/pla_de.yaml")

evaluator = ZicoProxy(
    loss_fn=torch.nn.CrossEntropyLoss(),
    repetitions=None
)
# data_getter = instantiate(cfg.data)
# dataloader = data_getter.load()

# model = efficientnet_b0(num_classes=10)
default_chromo = instantiate(
    cfg.chromosome,
    chromo=[2, 4, 4, 1, 1, 1, 1, 6, 7, 1]
)
chromosomes = [default_chromo] * (cfg.execute.n_cells)
model_hparams = cfg.model
model = Network(
    chromosomes,
    model_hparams.network_channels,
    model_hparams.strides,
    model_hparams.dropout,
    model_hparams.num_classes,
    model_hparams.last_channels,
)
model = model.classifier

model = Block(
    4, 24, 40, 1, [6, 7, 1]
)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

device = torch.device("cuda")
repetitions = 1000

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
dummy_input = torch.rand(1, 24, 32, 32).to(device)
model.to(device)
model.eval()
# warm up
for _ in range(100):
    _ = model(dummy_input)

timings = []
with torch.no_grad():
    for rep in tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings.append(curr_time)

    mean_latency = sum(timings) / repetitions

print(mean_latency)

score_list = []
# for _ in range(50):
#     score = evaluator.get_zico(model, dataloader, "cuda")
#     score_list.append(score)

# print(sum(score_list) / 50)
