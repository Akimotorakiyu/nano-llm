import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch import nn

from dataloader import NanoDataLoader
from src.model import NanoLLM


class NanoTrainer:
    def __init__(self, model: NanoLLM, dataloader: NanoDataLoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def save_model(self, epoch, batch_idx):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving model checkpoint at epoch {epoch + 1}...")
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / f"nano_llm_epoch_{epoch + 1}_{batch_idx + 1}.pth",
        )
        torch.save(self.model.state_dict(), checkpoint_dir / f"nano_llm_epoch_last.pth")

    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)

        self.optimizer.zero_grad()
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, epochs):
        print("Training Configuration:")
        print("epochs:", epochs)
        print("batch_size:", self.dataloader.batch_size)

        print("Starting training...")
        for epoch in range(epochs):
            epoch_total_loss = 0
            for batch_idx, batch in enumerate(self.dataloader):
                inputs, targets = batch
                loss = self.train_step(inputs, targets)

                print(
                    f"epoch {epoch + 1}/{epochs}, batch {batch_idx + 1}/{len(self.dataloader)}, Batch Loss: {loss:.4f}"
                )

                epoch_total_loss += loss

                if (batch_idx + 1) % 10 == 0:
                    print(f"saving model checkpoint...")
                    self.save_model(epoch, batch_idx)

            avg_loss = epoch_total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
