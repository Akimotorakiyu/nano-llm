import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch import nn

from dataloader import NanoDataLoader
from src.model import NanoLLM


class NanoTrainer:
    def __init__(self, model: NanoLLM, dataloader: NanoDataLoader):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, inputs, targets):
        outputs = self.model(inputs)

        self.optimizer.zero_grad()
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, epochs):
        for epoch in range(epochs):
            epoch_total_loss = 0
            for batch in self.dataloader:
                inputs, targets = batch
                loss = self.train_step(inputs, targets)

                print(f"Batch Loss: {loss:.4f}")

                epoch_total_loss += loss

            avg_loss = epoch_total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
