import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T


class DatasetCifar100(Dataset):
    def __init__(self, dset, num_classes):
        self.dset = dset
        self.num_classes = num_classes

    def __getitem__(self, index):
        x, y = self.dset[index]
        ones = torch.ones(self.num_classes)
        ones[y] = 1.

        return x, ones

    def __len__(self):
        return len(self.dset)


class DataLoaderforSearchGetter:
    def __init__(
        self,
        data_dir,
        batch_size,
        image_size,
        crop_size,
        n_batches
    ):
        self.data_dir = data_dir
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.image_size = image_size
        self.crop_size = crop_size

    def transforms(self):
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(
                self.image_size,
                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return transform

    def dataset(self):
        dset = torchvision.datasets.CIFAR100(
            self.data_dir,
            train=True,
            transform=self.transforms(),
            download=True
        )
        return DatasetCifar100(dset, num_classes=100)
        # return dset

    def load(self):
        dloader = DataLoader(
            self.dataset(),
            batch_size=self.batch_size,
            shuffle=True
        )

        batches = []
        for i, batch in enumerate(dloader):
            if i < self.n_batches:
                batches.append(batch)

        return batches
