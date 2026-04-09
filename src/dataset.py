import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class NanoDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = (
            self.tokenizer(
                str(sample["text"]),
                add_special_tokens=False,
                max_length=self.max_length - 2,
                truncation=True,
            )
            .input_ids.squeeze(0)
            .tolist()
        )
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 正确的 labels: 预测下一个 token，需要 shift 一个位置
        input_ids = tokens[:-1]  # 去掉最后一个 token
        labels = tokens[1:]      # 去掉第一个 token (BOS)，shifted

        # padding
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - 1 - len(input_ids))
        labels = labels + [-100] * (self.max_length - 1 - len(labels))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels


if __name__ == "__main__":
    from .tokenizer.tokenizer import NanoTokenizer

    tokenizer = NanoTokenizer()
    dataset = NanoDataset("./data/pretrain_t2t_mini.jsonl", tokenizer, max_length=512)
    print("Dataset length:", len(dataset))
    print("First item tokenized:", dataset[0])
