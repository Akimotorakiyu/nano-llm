from .dataset import NanoDataSet
import torch


def collate_fn(batch):
    """
    自定义 collate_fn：batch_size=1 时直接返回单个样本，不做堆叠
    """
    # batch 是 [(input_seq, target_seq)]，只有一个元素
    return batch[0]


class NanoDataLoader(torch.utils.data.DataLoader):
    """
    新架构数据加载器：加载完整序列 (input_seq, target_seq)
    每个 batch 是一条完整文本序列
    """

    def __init__(self, dataset: NanoDataSet, batch_size: int = 1, shuffle: bool = True):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
