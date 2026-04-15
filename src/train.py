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
        total_loss = 0.0
        num_batches = 0

        for e in range(epoch):
            for batch in dataloader:
                x, y = batch
                loss = self.train_step(x, y)
                total_loss += loss.item()
                num_batches += 1
                print(f"epoch {e + 1}/{epoch} - loss: {loss}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

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