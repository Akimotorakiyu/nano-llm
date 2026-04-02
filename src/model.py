from torch import nn
import torch


class NanoConfig:
    def __init__(
        self, n_layers=8, embeding_dim=512, attention_dim=768, vocab_size=6400
    ):
        self.n__layers = n_layers
        self.embeding_dim = embeding_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size


# [N,D] -> [N,D]
class NanoAttention(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.attention_dim = config.attention_dim
        self.Q = nn.Linear(config.attention_dim, config.attention_dim)
        self.K = nn.Linear(config.attention_dim, config.attention_dim)
        self.V = nn.Linear(config.attention_dim, config.attention_dim)

    def forward(self, x):
        # [N,D] @ [D,D] -> [N,D]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # [N,D] @ [D,N] -> [N,N]
        attention_socres = Q @ K.transpose(-2, -1) / (self.attention_dim**0.5)

        # 生成一个上三角矩阵，遮蔽未来信息
        mask = torch.triu(torch.ones_like(attention_socres), diagonal=1).bool()
        attention_socres = attention_socres.masked_fill(mask, float('-inf'))


        attention_qk = nn.functional.softmax(attention_socres, dim=-1)

        # [N,N] @ [N,D] -> [N,D]
        output = attention_qk @ V

        return output


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        """
        Args:
            dim: 输入维度 (例如 4096)
            hidden_dim: 中间层维度 (通常是 dim 的倍数，如 11008)
        """
        super().__init__()
        # w1: 主分支 (用于 Swish)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # w3: 门控分支 (用于线性门控)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # w2: 输出投影
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """
        公式: SwiGLU(x) = (Swish(xW1) ⊗ xW3) W2
        """
        # 1. 分别计算两个分支
        # F.silu 等同于 Swish(x) = x * sigmoid(x)
        swish_branch = nn.functional.silu(self.w1(x))
        gate_branch = self.w3(x)

        # 2. 逐元素相乘 (门控机制)
        hidden = swish_branch * gate_branch

        # 3. 输出投影
        return self.w2(hidden)


# [N,D] -> [N,D]
class NanoFeedForward(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.fnn = SwiGLU(config.attention_dim, config.attention_dim * 4)

    def forward(self, x):
        return self.fnn(x)


class NanoTransformerBlock(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.attention_rms = nn.RMSNorm(config.attention_dim)
        self.attention = NanoAttention(config)
        self.feed_forward_rms = nn.RMSNorm(config.attention_dim)
        self.feed_forward = NanoFeedForward(config)

    def forward(self, x):
        x = self.attention(self.attention_rms(x)) + x
        x = self.feed_forward(self.feed_forward_rms(x)) + x
        return x


class NanoEmbending(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.token_embeding = nn.Embedding(config.vocab_size, config.embeding_dim)
        self.projection = nn.Linear(config.embeding_dim, config.attention_dim)

    def forward(self, x):
        x = self.token_embeding(x)
        return self.projection(x)


class NanoOutput(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.rms = nn.RMSNorm(config.attention_dim)
        self.projection = nn.Linear(config.attention_dim, config.vocab_size)

    def forward(self, x):
        x = self.rms(x)
        return self.projection(x)


class NanoLLM(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.embending = NanoEmbending(config)
        self.layers = nn.ModuleList(
            [NanoTransformerBlock(config) for _ in range(config.n__layers)]
        )
        self.output = NanoOutput(config)

    def forward(self, x):
        x = self.embending(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
