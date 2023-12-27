import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision
import torchvision.transforms as T
from xautodl.datasets.DownsampledImageNet import ImageNet16


class DatasetImageNet16(Dataset):
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
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]

        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(
                self.image_size,
                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(self.crop_size, padding=2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return transform

    def dataset(self):
        dset = ImageNet16(
            self.data_dir,
            train=True,
            transform=self.transforms(),
            use_num_of_class_only=120
        )
        return DatasetImageNet16(dset, num_classes=120)
        # return dset

    def load(self):
        print(len(self.dataset()))
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
