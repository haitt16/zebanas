import torch
from tqdm import tqdm
import numpy as np


class DatabaseQueryEvaluator:
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.data = self.read_file(obj_path)

    def read_file(self, obj_path):
        data = torch.load(obj_path)
        return data

    def query(self, chromosome):
        key = tuple(chromosome.data)
        obj = self.data[key]
        return obj

    def __call__(
        self,
        cfg,
        dataloader,
        samples,
        chromosomes,
        search_index
    ):
        objs = []
        for chromo in samples:
            obj = self.query(chromo)
            objs.append(obj)

        return samples.set_obj(np.array(objs))
