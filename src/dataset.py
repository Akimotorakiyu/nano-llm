import datasets
import torch.utils.data.dataset

class NanoDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset = datasets.load_dataset(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]