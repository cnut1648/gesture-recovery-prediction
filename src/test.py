import os

import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W =  nn.Parameter(torch.randn(32, 2))
    def forward(self, x):
        return x @ self.W



class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        # self.layer = torch.nn.Linear(32, 2)
        self.layer = M()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)


        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=int(25e3),
            verbose=False,
        )
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "OneCycleLR",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_dict,
        }


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=7,
        gpus=-1,
        weights_summary=None,
        # precision=16
    )
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)


if __name__ == '__main__':
    run()