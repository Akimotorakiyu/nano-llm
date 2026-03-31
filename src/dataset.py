import json
import torch
from tokenizer.tokenizer import NanoTokenizer


class NanoDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer: NanoTokenizer, max_lines=None):
        self.tokenizer = tokenizer
        self.max_lines = max_lines
        self.dataset = self.load__data(path)

    def load__data(self, path):
        seq = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_lines and i >= self.max_lines:
                    break
                data = json.loads(line)
                text = data["text"]
                input_ids = self.map_text(text)
                seq.append(input_ids)
        self.seq = seq
        return seq

    def map_text(self, text):
        str = (
            self.tokenizer.mini_mind_tokenizer.eos_token
            + text
            + self.tokenizer.mini_mind_tokenizer.eos_token
        )
        input_ids = self.tokenizer.tokenizer(str)
        return input_ids.squeeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.seq[idx]

        input_ids, target_ids = item[:-1], item[1:]

        return input_ids, target_ids


if __name__ == "__main__":
    tokenizer = NanoTokenizer()
    dataset = NanoDataset("./data/pretrain_t2t_mini.jsonl", tokenizer, max_lines=10)
    print("Dataset length:", len(dataset))
    print("First item tokenized:", dataset[0])
