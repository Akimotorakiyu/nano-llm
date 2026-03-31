from torch.utils.data import DataLoader
from dataset import NanoDataset
import torch



class NanoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=256, shuffle=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        target_ids = [item[1] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        # 这里的 padding_value=-100 是为了在计算损失时忽略填充部分的影响，具体取决于你的损失函数如何处理标签
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=-100)

        return input_ids, target_ids