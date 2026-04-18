import torch
from typing import List

startTokenId = 2
endTokenId = 4

startToken = torch.tensor([startTokenId])
endToken = torch.tensor([endTokenId])


class NanoDataSet(torch.utils.data.Dataset):
    def genSamples(self, text: str):
        samples: List[str] = []
        for i in range(1, len(text)):
            x = torch.cat([startToken, torch.tensor([ord(c)
                          for c in text[:i]])])
            y = torch.tensor([ord(c) for c in text[: i + 1]])
            samples.append((x, y))

        x = torch.cat([startToken, torch.tensor([ord(c) for c in text])])
        y = torch.cat([torch.tensor([ord(c) for c in text]), endToken])

        samples.append((x, y))
        return samples

    def __init__(self):
        super().__init__()
        self.samples: List[str] = []
        raw_samples = [
            "hello world!",
            "I'm a bot!",
            "Good night!",
            "Pandas are fragile (newborns weigh just 100g!) but remain one of conservations biggest success stories.",
            # 100 以内的加法
            # *[f"{i}+{j}={i+j}" for i in range(1, 10) for j in range(1, 10)],
            # 99 乘法表
            # *[f"{i}×{j}={i*j}" for i in range(1, 10) for j in range(1, 10)],
        ]
        for sample in raw_samples:
            samples = self.genSamples(sample)
            for s in samples:
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        return sample
