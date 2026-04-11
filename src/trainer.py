import time
from pathlib import Path

import swanlab
import torch
from torch import nn

from .dataloader import NanoDataLoader
from .model import NanoLLM, NanoConfig
from .optim import CAdamW

from torch.optim.lr_scheduler import CosineAnnealingLR


class IterTimer:
    def __init__(self):
        self._start = None

    def tick(self):
        if self._start is None:
            self._start = time.time()
            return 0.0
        elapsed = time.time() - self._start
        self._start = time.time()
        return 1.0 / elapsed if elapsed > 0 else 0.0


class NanoTrainer:
    def __init__(
        self,
        model: NanoLLM,
        dataloader: NanoDataLoader,
        config: NanoConfig = None,
        swanlab_project: str = "nano-llm",
        swanlab_experiment_name: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader

        # 统一配置管理 —— 最终最优超参
        self.train_config = {
            "optimizer": "CAdamW",
            "learning_rate": 5e-5,       # 从 1e-4 下调到 5e-5，更稳定
            "weight_decay": 1e-5,        # 标准稳定值
            "betas": (0.9, 0.999),       # 官方标准
            "batch_size": dataloader.batch_size,
        }
        if config is not None:
            self.train_config.update({
                "n_layers": config.n_layers,
                "embedding_dim": config.embedding_dim,
                "attention_dim": config.attention_dim,
                "vocab_size": config.vocab_size,
            })

        self.optimizer = CAdamW(
            self.model.parameters(),
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"],
            betas=self.train_config["betas"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_step = 0

        # 初始化 SwanLab
        swanlab.init(
            project=swanlab_project,
            experiment_name=swanlab_experiment_name,
            config=self.train_config,
            mode="local"
        )

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 用于优雅退出时保存进度
        self.current_epoch = 0
        self.current_batch = -1

    def save_model(self, epoch, batch_idx, loss=None):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving model checkpoint at epoch {epoch + 1}...")
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if loss is not None:
            checkpoint["loss"] = loss
        torch.save(
            checkpoint,
            checkpoint_dir / f"nano_llm_epoch_{epoch + 1}_{batch_idx + 1}.pth",
        )
        torch.save(checkpoint, checkpoint_dir / "nano_llm_last.pth")

    def load_checkpoint(self, checkpoint_path):
        """加载 checkpoint 进行断点续训练，文件不存在返回 None"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"], checkpoint["batch_idx"], checkpoint.get("loss")

    def train_step(self, inputs, targets):
        # 1. 必加：训练模式（开启Dropout/BatchNorm）
        self.model.train()
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)

        # 2. 必加：梯度清零（位置修正！）
        self.optimizer.zero_grad()

        # 展平
        logits = outputs.view(-1, outputs.size(-1))
        labels = targets.view(-1)

        # 损失
        loss = self.loss_fn(logits, labels)

        # 反向传播 + 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

        # 记录
        swanlab.log(
            {
                "train/loss": loss.item(),
                "train/accuracy": acc,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            },
            step=self.global_step,
        )
        self.global_step += 1

        return loss.item(), acc

    def train(self, epochs, resume_from="checkpoints/nano_llm_last.pth"):
        start_epoch = 0
        start_batch = -1

        result = self.load_checkpoint(resume_from)
        if result is not None:
            start_epoch, start_batch, last_loss = result
            if last_loss is not None:
                print(f"Last checkpoint loss: {last_loss:.4f}")

        print("Training Configuration:")
        print("epochs:", epochs)
        print("batch_size:", self.dataloader.batch_size)
        if start_epoch > 0 or start_batch >= 0:
            print(f"resuming from epoch {start_epoch + 1}, batch {start_batch + 1}")

        print("Starting training...")
        timer = IterTimer()
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_total_loss = 0
            num_batches = 0
            for batch_idx, batch in enumerate(self.dataloader):
                if epoch == start_epoch and batch_idx <= start_batch:
                    continue

                self.current_batch = batch_idx
                inputs, targets = batch
                loss, acc = self.train_step(inputs, targets)

                iter_per_sec = timer.tick()
                print(
                    f"epoch {epoch + 1}/{epochs}, batch {batch_idx + 1}/{len(self.dataloader)}, Batch Loss: {loss:.4f}, Accuracy: {acc:.4f}, iter/s: {iter_per_sec:.2f}"
                )

                epoch_total_loss += loss
                num_batches += 1

                if (batch_idx + 1) % 1000 == 0:
                    print("saving model checkpoint...")
                    self.save_model(epoch, batch_idx, loss=loss)

            # 3. 必加：更新学习率调度器
            self.scheduler.step()
            
            if num_batches > 0:
                avg_loss = epoch_total_loss / num_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                swanlab.log(
                    {
                        "epoch/avg_loss": avg_loss,
                        "epoch/num": epoch + 1,
                    },
                    step=self.global_step,
                )

        swanlab.finish()