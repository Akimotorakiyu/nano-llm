import torch
from typing import List, Tuple

startTokenId = 2
endTokenId = 4

startToken = torch.tensor([startTokenId])
endToken = torch.tensor([endTokenId])


class NanoDataSet(torch.utils.data.Dataset):
    """
    Demo 阶段数据集：每条样本是一个完整文本序列
    返回: (input_seq, target_seq)
    """

    def __init__(self):
        super().__init__()
        # sequences: 每条文本作为一个序列 [(input_seq, target_seq), ...]
        self.sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []

        raw_samples = [
            "hello world!",
            "I'm a bot!",
            "Good night!",
        ]

        for sample in raw_samples:
            input_seq, target_seq = self.genSequence(sample)
            self.sequences.append((input_seq, target_seq))

    def genSequence(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成完整序列
        返回: (input_seq, target_seq)
        input_seq:  (seq_len,) - 从 <start> 开始到倒数第二个token
        target_seq: (seq_len,) - 从第一个字符开始到 <end>
        """
        chars = [ord(c) for c in text]

        # 构建完整序列: <start> + text + <end>
        full_sequence = [startTokenId] + chars + [endTokenId]

        # input:  除最后一个token外的所有token
        # target: 除第一个token外的所有token
        input_seq = torch.tensor(full_sequence[:-1])   # (seq_len,)
        target_seq = torch.tensor(full_sequence[1:])   # (seq_len,)

        return input_seq, target_seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index]
