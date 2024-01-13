# import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from lightning.pytorch.core import LightningDataModule


class DatasetCifar10(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        x, y = self.dset[index]
        return x, y

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
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

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
        dset = CIFAR10(
            self.data_dir,
            train=True,
            transform=self.transforms(),
            download=True
        )
        return DatasetCifar10(dset)
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


class CIFAR10LightningModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        image_size,
        crop_size,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.crop_size = crop_size

    def get_transforms(self, stage):
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        if stage == "fit":
            transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Resize(
                    self.image_size,
                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        elif stage == "test":
            transform = T.Compose([
                T.Resize(
                    self.image_size,
                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            raise ValueError(
                "There is no transform implemented for this stage"
            )

        return transform

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        if stage == "fit":
            transform = self.get_transforms("fit")
            self.train_set = CIFAR10(
                self.data_dir, train=True, transform=transform
            )
            transform = self.get_transforms("test")
            self.val_set = CIFAR10(
                self.data_dir, train=False, transform=transform
            )
        elif stage == "test":
            transform = self.get_transforms("test")
            self.test_set = CIFAR10(
                self.data_dir,
                train=False,
                transform=transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            pin_memory=True
        )
