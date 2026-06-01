import math
import torch
from torch.nn.functional import silu
from dataclasses import dataclass


@dataclass
class NanoLLMConfig:
    vocab_size: int = 256  # 支持更多字符（如 × 等符号）
    hidden_dim: int = 256
    mem_len: int = 1024  # 状态矩阵的行数


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

    def forward(self, x: torch.Tensor, mem_state: torch.Tensor):
        # Nope 设计: Q来自mem_state, K/V来自输入x
        # x: (batch_size, seq_len, hidden_dim), mem_state: (batch_size, mem_len, hidden_dim)
        Q = self.q(mem_state)
        K = self.k(x)
        V = self.v(x)

        attention_scores = (Q @ K.transpose(-2, -1)) / self.dim_sqrt

        # Causal Mask (对于单token，mask全为0，即不看未来的信息)
        mask = torch.triu(
            torch.ones_like(attention_scores, dtype=torch.float32), diagonal=1
        )
        attention_scores = attention_scores.masked_fill(mask == 1, -1e9)

        output = torch.softmax(attention_scores, dim=-1) @ V

        return output  # (batch, mem_len, hidden_dim)


class SwiGLU(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        # 4 倍升维
        self.intermediate_dim = self.config.hidden_dim * 4

        # 升维投影
        self.w1 = torch.nn.Linear(
            self.config.hidden_dim, self.intermediate_dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            self.config.hidden_dim, self.intermediate_dim, bias=False
        )
        # 降维投影
        self.w2 = torch.nn.Linear(
            self.intermediate_dim, self.config.hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor):
        """SwiGLU: swish(x @ w1) * (x @ w3) @ w2"""
        return self.w2(silu(self.w1(x)) * self.w3(x))


class NanoFeedForward(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.swiGLU = SwiGLU(config)

    def forward(self, x: torch.Tensor):
        return self.swiGLU(x)


class NanoTransformerBlock(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.attention = NanoSelfAttention(config)
        self.ffn = NanoFeedForward(config)

        self.attention_rms = torch.nn.RMSNorm(config.hidden_dim)
        self.ffn_rms = torch.nn.RMSNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, mem_state: torch.Tensor):
        mem_state = mem_state + self.attention(self.attention_rms(x), self.attention_rms(mem_state))
        mem_state = mem_state + self.ffn(self.ffn_rms(mem_state))
        return mem_state


class NanoLLM(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config

        self.embedding = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_dim
        )

        self.nanoTransformerBlock = torch.nn.ModuleList([
            NanoTransformerBlock(config) for _ in range(8)
        ])

        self.output = torch.nn.Linear(
            self.config.hidden_dim, self.config.vocab_size)

        self.init_mem_state = torch.nn.Parameter(torch.zeros(self.config.mem_len, self.config.hidden_dim),False)

    def forward(self, x: torch.Tensor, mem_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len) - 当前输入token
        # mem_state: (batch_size, mem_len, hidden_dim) - 当前状态，None时用self.init_mem_state矩阵生成初始状态

        # 嵌入当前输入token: (batch, seq_len) -> (batch, seq_len, hidden)
        x_embedded = self.embedding(x)  # (batch, seq_len, hidden)

        if mem_state is None:
            # self.init_mem_state: (mem_len, hidden) -> 扩展为 (batch, mem_len, hidden)
            mem_state = self.init_mem_state.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, mem_len, hidden)

        # Nope风格：x提供K/V，mem_state提供Q，输出更新后的状态
        for block in self.nanoTransformerBlock:
            mem_state = block(x_embedded, mem_state)

        # 基于状态mem_state预测下一个token: 将3D状态汇聚后映射到vocab_size
        mem_state_flat = mem_state.mean(dim=1)  # (batch, hidden_dim)
        y = self.output(mem_state_flat)  # (batch_size, vocab_size)
        return y, mem_state
