import os, random
import pickle
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.data.MCDataModule import SeqAttnData



class CombinedData(Dataset):
    def __init__(self, gesture: dict):
        self.gesture = gesture
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
    
class MCCombineDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str,
        batch_size: int, num_workers: int,
        k_fold: int,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dataset = {}
        self.k_fold = k_fold
    
    def setup(self, stage: Optional[str] = None):
        seq = {}
        for split in ['train', 'valid', 'test']:
            # file = os.path.join(self.data_dir, f"k0", f"{split}_processed.pkl")
            file = os.path.join(self.data_dir, f"k0", f"{split}_combined.pkl")
            assert os.path.exists(file)

            with open(file, "rb") as f:
                seq = {**seq, **pickle.load(f)}
            
        assert len(seq) == 50
        keys = list(seq.keys())
        random.seed(self.k_fold)
        random.shuffle(keys)
        traink, testk = keys[:40], keys[40:]
        train, test = {}, {}
        for k, v in seq.items():
            if k in traink:
                train = {**train, k:v}
            else:
                test = {**test, k:v}
        self.dataset["train"] = CombinedData(train)
        self.dataset["valid"] = CombinedData(test)
    
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