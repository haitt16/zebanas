from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


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
        dset = torchvision.datasets.CIFAR10(
            self.data_dir,
            train=True,
            transform=self.transforms(),
            download=True
        )
        return dset

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
