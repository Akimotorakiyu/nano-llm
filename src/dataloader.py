from torch.utils.data import DataLoader

from .dataset import NanoDataset

class NanoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True
        )


if __name__ == "__main__":
    from .tokenizer.tokenizer import NanoTokenizer

    tokenizer = NanoTokenizer()
    dataset = NanoDataset("./data/pretrain_t2t_mini.jsonl", tokenizer, max_length=512)
    dataloader = NanoDataLoader(dataset, batch_size=256, shuffle=True)

    for input_ids, target_ids in dataloader:
        print("Input IDs shape:", input_ids.shape)
        print("Target IDs shape:", target_ids.shape)
        break
