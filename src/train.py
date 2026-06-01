from .model import NanoLLM
from .dataloader import NanoDataLoader
import torch
from pathlib import Path


class Train:
    def __init__(self, model: NanoLLM):
        self.config = model.config
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters(), 5e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor, mem_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单步训练
        x: (batch_size,) - 输入token
        y: (batch_size,) 或标量 - 目标token（下一个token）
        mem_state: (batch_size, mem_len, hidden_dim) - 当前记忆状态，None表示重置状态
        """
        model = self.model
        model.zero_grad()

        # 前向传播，获取预测和更新后的状态
        logits, new_mem_state = model(x, mem_state)  # logits: (batch_size, vocab_size)

        # 确保 y 是 0D 或 1D（CrossEntropyLoss 要求）
        y = y.view(-1)  # 展平为 (batch_size,)

        # 计算损失（预测下一个token）
        loss: torch.Tensor = self.criterion(logits, y)
        loss.backward()
        self.optim.step()

        return loss, new_mem_state

    def train(self, dataloader: NanoDataLoader, epoch: int):
        """
        训练流程：
        - 每个 batch 是一条完整文本
        - batch 开始时重置 mem_state
        - batch 内部逐 token 训练，传递 mem_state
        """
        self.model.train()

        for e in range(epoch):
            epoch_loss = 0.0
            epoch_min_loss = float('inf')
            epoch_max_loss = 0.0
            epoch_losses = []
            num_tokens = 0

            for batch in dataloader:
                input_seq, target_seq = batch  # (seq_len,), (seq_len,)

                # 每条文本开始时重置 mem_state
                # mem_state 是模型内部记忆状态，形状 (batch, mem_len, hidden_dim)
                # 一条新文本开始时，状态应该从初始值开始，不应携带上一条文本的信息
                mem_state = None
                seq_loss = 0.0

                # 在文本内部逐 token 训练，传递 mem_state
                # 核心机制：模型根据当前输入 token 和 mem_state 预测下一个 token
                # 同时更新 mem_state，供下一个 token 预测使用
                # 这样 mem_state 会在文本内部不断演化，积累上下文信息
                for t in range(len(input_seq)):
                    x = input_seq[t:t+1]      # (1,) - 当前输入token
                    y = target_seq[t]         # 标量 - 目标token（下一个token）

                    # 训练一步：前向传播计算 loss，反向传播更新参数
                    # 返回的 mem_state 会在下一个 token 的预测中继续使用
                    # .detach() 切断梯度流，避免反向传播通过整个序列历史
                    loss, mem_state = self.train_step(x, y, mem_state)
                    mem_state = mem_state.detach()

                    loss_item = loss.item()
                    seq_loss += loss_item
                    epoch_losses.append(loss_item)
                    epoch_min_loss = min(epoch_min_loss, loss_item)
                    epoch_max_loss = max(epoch_max_loss, loss_item)
                    num_tokens += 1

                epoch_loss += seq_loss

            # 计算统计信息
            avg_loss = epoch_loss / num_tokens if num_tokens > 0 else 0.0
            if len(epoch_losses) > 1:
                variance = sum((loss - avg_loss) ** 2 for loss in epoch_losses) / len(epoch_losses)
                std_loss = variance ** 0.5
                cv_loss = std_loss / avg_loss if avg_loss != 0 else 0.0
            else:
                std_loss = 0.0
                cv_loss = 0.0

            print(f"epoch {e + 1}/{epoch} - tokens:{num_tokens} min:{epoch_min_loss:.4f} avg:{avg_loss:.4f} max:{epoch_max_loss:.4f} std:{std_loss:.4f} cv:{cv_loss:.4f}")

        # 保存检查点
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss.item() if 'loss' in dir() else 0.0,
            "avg_loss": avg_loss if 'avg_loss' in dir() else 0.0,
        }

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / "last.pt")
