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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        model = self.model
        model.zero_grad()
        output: torch.Tensor = model(x)

        loss: torch.Tensor = self.criterion(
            output.view(-1, output.size(-1)), y.view(-1)
        )
        loss.backward()
        self.optim.step()
        return loss

    def train(self, dataloader: NanoDataLoader, epoch: int):
        self.model.train()
        num_batches = 0

        for e in range(epoch):
            epoch_loss = 0.0
            epoch_min_loss = 100.0
            epoch_max_loss = 0.0
            epoch_losses = []
            for batch in dataloader:
                x, y = batch
                loss = self.train_step(x, y)
                loss_item = loss.item()
                epoch_loss += loss_item
                epoch_min_loss = min(epoch_min_loss, loss_item)
                epoch_max_loss = max(epoch_max_loss, loss_item)
                epoch_losses.append(loss_item)
                num_batches += 1

            avg_loss = epoch_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
            # 计算标准差和变异系数
            if len(epoch_losses) > 0:
                variance = sum((loss - avg_loss) ** 2 for loss in epoch_losses) / len(epoch_losses)
                std_loss = variance ** 0.5
                cv_loss = std_loss / avg_loss if avg_loss != 0 else 0.0
            else:
                std_loss = 0.0
                cv_loss = 0.0
            print(f"epoch {e + 1}/{epoch} - min:{epoch_min_loss:.4f} avg:{avg_loss:.4f} max:{epoch_max_loss:.4f} std:{std_loss:.4f} cv:{cv_loss:.4f}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss.item(),
            "avg_loss": avg_loss,
        }

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / "last.pt")