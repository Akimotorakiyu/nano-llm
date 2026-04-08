import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch import nn

from src.dataloader import NanoDataLoader
from src.model import NanoLLM


class NanoTrainer:
    def __init__(self, model: NanoLLM, dataloader: NanoDataLoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        # 默认 -100 用于忽略计算损失的 token ID
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 用于优雅退出时保存进度
        self.current_epoch = 0
        self.current_batch = -1

    def save_model(self, epoch, batch_idx):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving model checkpoint at epoch {epoch + 1}...")
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
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
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"], checkpoint["batch_idx"]

    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)

        self.optimizer.zero_grad()

        # 展平
        logits = outputs.view(-1, outputs.size(-1))
        labels = targets.view(-1)

        # 损失
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

        return loss.item(), acc

    def train(self, epochs, resume_from="checkpoints/nano_llm_last.pth"):
        """训练模型

        Args:
            epochs: 总训练轮数
            resume_from: checkpoint 路径，文件不存在则从头训练
        """
        start_epoch = 0
        start_batch = -1

        result = self.load_checkpoint(resume_from)
        if result is not None:
            start_epoch, start_batch = result

        print("Training Configuration:")
        print("epochs:", epochs)
        print("batch_size:", self.dataloader.batch_size)
        if start_epoch > 0 or start_batch >= 0:
            print(f"resuming from epoch {start_epoch + 1}, batch {start_batch + 1}")

        print("Starting training...")
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_total_loss = 0
            for batch_idx, batch in enumerate(self.dataloader):
                # 跳过已训练的 batch
                if epoch == start_epoch and batch_idx <= start_batch:
                    continue

                self.current_batch = batch_idx
                inputs, targets = batch
                loss, acc = self.train_step(inputs, targets)

                print(
                    f"epoch {epoch + 1}/{epochs}, batch {batch_idx + 1}/{len(self.dataloader)}, Batch Loss: {loss:.4f}, Accuracy: {acc:.4f}"
                )

                epoch_total_loss += loss

                if (batch_idx + 1) % 1000 == 0:
                    print("saving model checkpoint...")
                    self.save_model(epoch, batch_idx)

            avg_loss = epoch_total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
