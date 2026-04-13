from torch import nn
import torch
from mambapy.mamba import MambaBlock, MambaConfig


class NanoConfig:
    def __init__(
        self, n_layers=8, embedding_dim=512, attention_dim=768, vocab_size=6400,
        ssm_d_state=1, ssm_d_conv=1, ssm_expand_factor=1
    ):
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand_factor = ssm_expand_factor


class NanoSSSM(nn.Module):
    """使用 mambapy 的 Mamba Block"""

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.attention_dim)

        # 创建 MambaConfig
        mamba_config = MambaConfig(
            d_model=config.attention_dim,
            n_layers=1,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand_factor=config.ssm_expand_factor,
            pscan=True,
        )
        self.ssm = MambaBlock(mamba_config)

    def forward(self, x):
        # x: [B, N, D]
        x = self.ssm(self.norm(x))
        return x


class NanoEmbedding(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.attention_dim, bias=False)

    def forward(self, x):
        x = self.token_embedding(x)
        return self.projection(x)


class NanoOutput(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.rms = nn.RMSNorm(config.attention_dim)
        self.projection = nn.Linear(config.attention_dim, config.vocab_size, bias=True)

    def forward(self, x):
        x = self.rms(x)
        return self.projection(x)


class NanoLLM(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.embedding = NanoEmbedding(config)

        block = NanoSSSM(config)

        self.layers = nn.ModuleList([block for _ in range(config.n_layers)])
        self.output = NanoOutput(config)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
