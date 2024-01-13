import torch
import numpy as np
from tqdm import tqdm
import collections

from zebanas.data.vision.cifar10 import CIFAR10LightningModule
from zebanas.tasks.classification import NetworkModule
from zebanas.spaces.model import Gecco2024Network, Gecco2024Network2
from zebanas.criterions import MultitaskLossFunction
from zebanas.metrics.classifcation import Accuracy

datamodule = CIFAR10LightningModule(
    "/home/haitt/workspaces/data/vision/cifar10", 1024, 32, 32
)
datamodule.setup("fit")

model = Gecco2024Network(
    chromos=[
        [6, 6, 4, 4, 4, 3, 3, 3],
        [6, 6, 4, 6, 4, 4, 4, 3],
        [3, 6, 3, 3, 6, 3, 3, 3],
        [6, 5, 3, 3, 3, 3, 3, 3],
        [5, 6, 4, 4, 4, 3, 6, 3]
    ],
    network_channels=[16, 24, 48, 80, 128, 192],
    strides=[1, 1, 2, 1, 2, 1],
    dropout=0.1,
    num_classes=10,
    last_channels=1280
)
model2 = Gecco2024Network2(
    chromos=[
        [6, 6, 4, 4, 4, 3, 3, 3],
        [6, 6, 4, 6, 4, 4, 4, 3],
        [3, 6, 3, 3, 6, 3, 3, 3],
        [6, 5, 3, 3, 3, 3, 3, 3],
        [5, 6, 4, 4, 4, 3, 6, 3]
    ],
    network_channels=[16, 24, 48, 80, 128, 192],
    strides=[1, 1, 2, 1, 2, 1],
    dropout=0.1,
    num_classes=10,
    last_channels=1280
)
model = NetworkModule.load_from_checkpoint(
    "zebanas/checkpoints/cifar10/cifar10-epoch=187-val_score=0.4622.ckpt",
    model=model,
    loss_fn=MultitaskLossFunction(10),
    metric_fn=Accuracy()
)
model2 = NetworkModule.load_from_checkpoint(
    "zebanas/checkpoints/cifar10/cifar10-epoch=165-val_score=0.9282.ckpt",
    model=model2,
    loss_fn=MultitaskLossFunction(10),
    metric_fn=Accuracy()
)
model.eval()
model.cuda()
model2.eval()
model2.cuda()

# storage = [[], []]
# preds = []
# for x, y in tqdm(datamodule.train_dataloader()):
#     with torch.no_grad():
#         y_hat, e = model(x.cuda())
#         storage[0].append(e.cpu())
#         storage[1].append(y)
#         preds.append(y_hat.cpu())

# storage = [
#     torch.cat(storage[0]).numpy(),
#     torch.cat(storage[1]).numpy()
# ]
# preds = torch.cat(preds).numpy()
# torch.save({
#     "storage": storage,
#     "preds": preds
# }, "data_storage.pt")

data_storage = torch.load("data_storage.pt")
storage = data_storage["storage"]

datamodule.setup("test")

# predict

embs = []
p_cls_list = []
label_list = []
p_cls_2_list = []

for x, y in tqdm(datamodule.test_dataloader()):
    with torch.no_grad():
        p_cls, e = model(x.cuda())
        e = e.cpu().squeeze().numpy()
        embs.append(e)
        p_cls = p_cls.cpu().squeeze().numpy()
        p_cls_list.append(p_cls)
        label_list.append(y)

        p_cls = model2(x.cuda())
        p_cls_2_list.append(p_cls.cpu().squeeze().numpy())

embs = np.concatenate(embs)
p_cls_list = np.concatenate(p_cls_list)
label_list = np.concatenate(label_list)
p_cls_2_list = np.concatenate(p_cls_2_list)

alpha = 0.7
K = 32

cnt = 0
for e, p_cls, y, p_cls_2 in tqdm(zip(embs, p_cls_list, label_list, p_cls_2_list), total=embs.shape[0]):
    # knearest
    distance = np.sqrt(np.sum((storage[0] - e)**2, axis=1))

    si = np.argsort(distance)[:K]
    y_neig = storage[1][si]
    counter = collections.Counter(y_neig)
    p_knn = np.zeros(10)
    for k, v in counter.items():
        p_knn[k] = v

    p_knn = p_knn / np.sum(p_knn)
    p = alpha * p_knn + (1 - alpha) * p_cls

    label = np.argmax(p)
    label2 = np.argmax(p_cls_2)

    if int(label) == int(y) or int(label2) == int(y):
        cnt += 1

print(cnt)
