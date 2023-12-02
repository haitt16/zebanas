import torchvision

train_set = torchvision.datasets.ImageNet(
    ".", train=True, download=True
)

valid_set = torchvision.datasets.ImageNet(
    ".", train=False, download=True
)
