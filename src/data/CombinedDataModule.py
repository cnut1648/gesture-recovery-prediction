import os
from pathlib import Path
import pickle
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.data.SeqAttnDataModule import SeqAttnData

class CombinedData(Dataset):
    def __init__(self, gesture: dict, clinical: np.array):
        self.gesture = gesture
        self.clinical = clinical
        self.idx2caseid = {
            i: case
            for i, case in enumerate(self.gesture)
        }

    def __getitem__(self, idx: int):
        case = self.idx2caseid[idx]
        data = self.gesture[case]
        # clinical = self.clinical.query("`Case ID` == @case")\
        #                 .drop(['Case ID'], axis=1)\
        #                 .values[0]
        return case, data['L']['gesture'], data['R']['gesture'], data['label'], data['ft_transformer'][0]
    
    def __len__(self):
        return len(self.gesture)
    
    @staticmethod
    def collate_fn(batch):
        cbatch, gbatch = [], []
        for case, L, R, label, clinical in batch:
            cbatch.append(clinical)
            gbatch.append([case, L, R, label])
        gesture_output = SeqAttnData.collate_fn(gbatch)
        return {
            **gesture_output,
            **{"clinical": torch.from_numpy(np.stack(cbatch))}
        }

class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str, raw_dir: str,
        batch_size: int, num_workers: int,
        k_fold: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.raw_dir = Path(raw_dir)
        self.k_fold = k_fold
        self.dataset = {}
    
    def setup(self, stage: Optional[str] = None):
        for split in ['train', 'valid', 'test']:
            file = os.path.join(self.data_dir, f"k{self.k_fold}", f"{split}_combined.pkl")
            assert os.path.exists(file), f"{file} doesn't exist!"

            with open(file, "rb") as f:
                gesture = pickle.load(f)
            
            if split == "test":
                df = pd.read_excel(
                    self.raw_dir / "Dissection Gestures_Clinical Data (new cases).xlsx",
                    engine='openpyxl'
                )
                # only use the common 10 cases
                df = df[df['Case ID'].isin(gesture.keys())]
                df['CCI'][df['CCI'].isna()] = 7
            else:
                df = pd.read_excel(
                    self.raw_dir / "Dissection Gesture_Clinical Data.xlsx",
                    engine='openpyxl'
                )
            df['Nerve Sparing'] = df['Nerve Sparing'].apply(
                lambda val: 1 if val == 'full' else 0
            )
            clinical = df.drop(["Surgeon ID", "Pre-Op SHIM",], axis=1)
            # filter to only ids within this split
            clinical = clinical[clinical["Case ID"].isin(gesture.keys())]

            self.dataset[split] = CombinedData(gesture, clinical)
    
    def get_loader(self, split) -> DataLoader:
        return DataLoader(
            self.dataset[split], batch_size=self.batch_size,
            collate_fn=CombinedData.collate_fn, drop_last=False,
            num_workers=self.num_workers, pin_memory=True,
            shuffle=(split=="train")
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.get_loader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_loader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.get_loader("test")