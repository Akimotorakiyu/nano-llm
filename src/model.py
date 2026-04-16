import torch
from torch.nn.functional import silu
from dataclasses import dataclass


@dataclass
class NanoLLMConfig:
    vocab_size: int = 128
    hidden_dim: int = 256


class NanoSelfAttention(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.dim_sqrt = self.config.hidden_dim**0.5

        self.q = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.k = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.v = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor):
        # Nope 设计
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention_scores = (Q @ K.transpose(-2, -1)) / self.dim_sqrt

        # 标准 Causal Mask（保留下三角，屏蔽上三角）
        mask = torch.triu(
            torch.ones_like(attention_scores, dtype=torch.float32), diagonal=1
        )
        attention_scores = attention_scores.masked_fill(mask == 1, -1e9)

        output = torch.softmax(attention_scores, dim=-1) @ V

        return output


class NanoFeedForward(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.w1 = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.w2 = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor):
        """SwiGLU: swish(x @ w1) * (x @ w3) @ w2"""
        return self.w2(silu(self.w1(x)) * (self.w3(x)))


class NanoTransformerBlock(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.attention = NanoSelfAttention(config)
        self.ffn = NanoFeedForward(config)

        self.attention_rms = torch.nn.RMSNorm(config.hidden_dim)
        self.ffn_rms = torch.nn.RMSNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.attention_rms(x))
        x = x + self.ffn(self.ffn_rms(x))
        return x


class NanoLLM(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.embedding = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_dim
        )

        self.nanoTransformerBlock = torch.nn.ModuleList(
            [NanoTransformerBlock(config) for x in range(1)]
        )
        self.output = torch.nn.Linear(
            self.config.hidden_dim, self.config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.embedding(y)

        for block in self.nanoTransformerBlock:
            y = block(y)

        y = self.output(y)
        return y
