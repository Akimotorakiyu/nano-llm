import datasets
import torch

class NanoDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.load__data(path)

    def load__data(self, item):
        dataset = datasets.load_dataset(path)
        self.seq = self.map_text_arr(dataset[item]["text"])

    def map_text(self, text):
        str = self.tokenizer.eos_token + text + self.tokenizer.eos_token
        input_ids = self.tokenizer.tokenizer(str)
        return input_ids
    
    def map_text_arr(self, text_arr):

        seq = []

        for t in text_arr:
            str = self.tokenizer.eos_token + t + self.tokenizer.eos_token
            input_ids = self.map_text(str)
            seq.append(input_ids)
        
        return seq


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.seq[idx]

        input_ids,target_ids = item[:-1], item[1:]

        return input_ids, target_ids

if __name__ == "__main__":
    from tokenizer.tokenizer import NanoTokenizer
    tokenizer = NanoTokenizer()
    dataset = NanoDataset("./data/sample.json", tokenizer)
    print("Dataset length:", len(dataset))
    print("First item tokenized:", dataset[0])