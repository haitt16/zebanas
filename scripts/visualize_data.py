import torch

from zebanas.tasks.classification import NetworkModule
from zebanas.spaces.model import Gecco2024Network
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from zebanas.criterions.am_softmax import AdMSoftmaxLoss
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy.linalg import norm


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std),
])

dataset = CIFAR10(
    "/home/haitt/workspaces/data/vision/cifar10",
    train=True, transform=transform
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
test_set = CIFAR10(
    "/home/haitt/workspaces/data/vision/cifar10",
    train=False, transform=transform
)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True)

model = Gecco2024Network(
    chromos=[
        [1, 6, 4, 3, 6, 4, 4, 6],
        [2, 6, 4, 4, 4, 4, 4, 4],
        [2, 6, 3, 3, 3, 6, 3, 3],
        [2, 6, 3, 3, 3, 4, 3, 4],
        [2, 6, 3, 3, 3, 6, 6, 3]
    ],
    network_channels=[16, 24, 48, 80, 128, 196],
    strides=[2, 2, 2, 2, 2, 1],
    dropout=0.1,
    num_classes=1000,
    last_channels=1280
)
loss_fn = AdMSoftmaxLoss(1000, 10)
model = NetworkModule.load_from_checkpoint(
    "zebanas/checkpoints/cifar10/cifar10-epoch=33-valid_loss=1.24.ckpt",
    model=model,
    loss_fn=loss_fn, metric_fn=None
)
model.eval()
model.cuda()

embedding_list = []
label_list = []

# with torch.no_grad():
#     for x, y in tqdm(dataloader):
#         emb = model(x.cuda())
#         embedding_list.append(emb.detach().cpu())
#         label_list.append(y)
    
#     for x, y in tqdm(test_loader):
#         emb = model(x.cuda())
#         embedding_list.append(emb.detach().cpu())
#         label_list.append(y)

# embeddings = torch.cat(embedding_list).numpy()
# labels = torch.cat(label_list).numpy()


# tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, verbose=1)
# embeddings_2d = tsne.fit_transform(embeddings)



ckpt = torch.load("/home/haitt/workspaces/codes/nas/zebanas/zebanas/checkpoints/cifar10/embeddings_2d.pt")
embeddings = ckpt["embeddings"]
embeddings_2d = ckpt["embeddings2d"]
labels = ckpt["labels"]
print(np.sum(labels == 0)) 
torch.save({
    "embeddings": embeddings,
    "embeddings2d": embeddings_2d,
    "labels": labels
}, "zebanas/checkpoints/cifar10/embeddings_2d.pt")


embeddings_train = embeddings[:50000]
embeddings_2d_train = embeddings_2d[:50000]
labels_train = labels[:50000]

label2emb_dict = {}
for i in tqdm(range(10)):
    indexs = np.where(labels_train == i)[0]
    embs = embeddings_train[indexs]

    label2emb_dict[i] = embs

embeddings_test = embeddings[50000:]
embeddings_2d_test = embeddings_2d[50000:]
labels_test = labels[50000:]
label2emb_dict_test = {}
for i in tqdm(range(10)):
    indexs = np.where(labels_test == i)[0]
    embs = embeddings_test[indexs]

    label2emb_dict_test[i] = (embs, labels)


# colors = ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
# fig = plt.figure()
# # ax = plt.axes(projection='3d')

# for k, v in label2emb_dict.items():
#     plt.scatter(v[:, 0], v[:, 1], c=colors[k])
# plt.scatter(label2emb_dict_test[1][:, 0], label2emb_dict_test[1][:, 1], marker="^", c=colors[1])
# plt.show()

index = 0
cnt = 0
print(embeddings_train.shape)

for index in range(10):
    for sample in tqdm(label2emb_dict_test[index][0]):
        # print(sample.shape)
        distances = np.dot(embeddings_train, sample) / (norm(embeddings_train, axis=1)*norm(sample))
        # print(distances)

        indexs = np.argsort(distances)[-32:]
        k_labels = labels_train[indexs]
        values, counts = np.unique(k_labels, return_counts=True) 
        
        i = np.argmax(counts)
        l = values[i]
        if l == index:
            cnt += 1

print(cnt)