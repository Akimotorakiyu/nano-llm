from torch import nn
import torch


class NanoConfig:
    def __init__(
        self, n_layers=8, embedding_dim=512, attention_dim=768, vocab_size=6400
    ):
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size


# [N,D] -> [N,D]
class NanoAttention(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.attention_dim = config.attention_dim
        self.Q = nn.Linear(config.attention_dim, config.attention_dim,bias=False)

    def forward(self, x):
        # [N,D] @ [D,D] -> [N,D]
        Q = x
        K = Q
        V = x

        # [N,D] @ [D,N] -> [N,N]
        attention_scores = Q @ K.transpose(-2, -1) / (self.attention_dim**0.5)

        # 生成一个上三角矩阵，遮蔽未来信息

        # 1. 直接生成 float 类型的 mask，避免后续的 bool() 转换开销
        # torch.float32 也可以，但在 FP16 环境下要注意数值范围
        mask = torch.triu(torch.ones_like(attention_scores, dtype=torch.float32), diagonal=1)

        # 2. 使用 -1e9 代替 -torch.inf，防止 Softmax 出现 NaN
        attention_scores = attention_scores.masked_fill(mask == 1, -1e9)

        attention_qk = nn.functional.softmax(attention_scores, dim=-1)

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


class ResidualLinearMerge(nn.Module):
    """线性映射层：将 [B, N, D] 映射为 [B, 1, D]
    递归残差连接： output = self.linear0(output) + self.linear1(x[:, i:i+1, :])
    """
    def __init__(self, dim):
        super().__init__()
        self.linear0 = nn.Linear(dim, dim, bias=False)
        self.linear1 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        B, N, _ = x.shape
        output = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)  # [B, 1, D]
        for i in range(N):
            # output: [B, 1, D], x[:, i:i+1, :]: [B, 1, D]
            output = self.linear0(output) + self.linear1(x[:, i:i+1, :])
        return output


class NanoTransformerBlock(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.attention_rms = nn.RMSNorm(config.attention_dim)
        self.attention = NanoAttention(config)
        self.feed_forward_rms = nn.RMSNorm(config.attention_dim)
        self.feed_forward = NanoFeedForward(config)
        self.merge = ResidualLinearMerge(config.attention_dim)

    def forward(self, x):
        # x: [B, N, D]
        x_next = self.attention(self.attention_rms(x))
        x_next = self.feed_forward(self.attention_rms(x_next))

        # 将 x_next 的 [B, N, D] 映射为 [B, 1, D]
        x_next_merged = self.merge(x_next)

        # 移除首位 [B, 1, D]，在序列结尾拼接 [B, 1, D]
        output = torch.cat([x[:, 1:, :], x_next_merged], dim=-2)

        return output


class NanoEmbending(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        # Nope 设计, 直接使用 nn.Embedding 来处理输入的 token ID
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
        self.embending = NanoEmbending(config)
        
        block = NanoTransformerBlock(config)
        self.layers = nn.ModuleList(
            [ block for _ in range(1)]
        )
        self.output = NanoOutput(config)

    def forward(self, x):
        x = self.embending(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
