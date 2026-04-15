
from .dataset import NanoDataSet
import torch


class NanoDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: NanoDataSet, batch_size=1):
        super().__init__(
            dataset, batch_size=batch_size
        )
