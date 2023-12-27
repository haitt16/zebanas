from zebanas.data.vision.imagenet16 import DataLoaderforSearchGetter

train = DataLoaderforSearchGetter(
    '/home/haitt/workspaces/data/vision/imagenet16',
    2, 16, 16, 2
)
train = train.load()