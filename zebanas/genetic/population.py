import numpy as np
# from tqdm import tqdm
from hydra.utils import instantiate


class Population(np.ndarray):
    def __new__(cls, chromos=[]):
        if not isinstance(chromos, list):
            chromos = [chromos]
        return np.array(chromos).view(cls)

    def get_data(self):
        data = [ch.data for ch in self]
        return np.array(data)

    def get_obj(self):
        obj = [ch.obj for ch in self]
        return np.array(obj)

    def set_obj(self, objs):
        for i in range(self.shape[0]):
            self[i].obj = objs[i]
        return self

    @classmethod
    def create(cls, *args):
        return Population.__new__(cls, args)

    @classmethod
    def merge(cls, pop_list):
        return Population.__new__(
            cls,
            np.concatenate(pop_list).tolist()
        )

    @classmethod
    def new(cls, cfg, data):
        return Population.__new__(
            cls,
            [instantiate(cfg.chromosome, chromo=d) for d in data]
        )
