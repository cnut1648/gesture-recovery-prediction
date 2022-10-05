
import os
import pickle
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SeqAttnData(Dataset):
    def __init__(self, seq: dict):
        self.seq = seq
        self.idx2caseid = {
            i: case
            for i, case in enumerate(self.seq)
        }

    def __len__(self):
        return len(self.idx2caseid)

    def __getitem__(self, idx):
        case = self.idx2caseid[idx]
        data = self.seq[case]
        return case, data['L']['gesture'], data['R']['gesture'], data['label']

    @staticmethod
    def collate_fn(batch):
        # max seq len
        max_seq_len = 882
        max_len_L = max(
            len(L) for _, L, _, _ in batch
        )
        max_len_L = min(max_len_L, max_seq_len)
        input_ids_L = torch.zeros((len(batch), max_len_L))
        attn_mask_L = torch.zeros((len(batch), max_len_L))
        max_len_R = max(
            len(R) for _, _, R, _ in batch
        )
        max_len_R = min(max_len_R, max_seq_len)
        input_ids_R = torch.zeros((len(batch), max_len_R))
        attn_mask_R = torch.zeros((len(batch), max_len_R))
        labels = torch.zeros((len(batch),))
        cases = []
        for i, (case, L, R, label) in enumerate(batch):
            # shift right and add 3 (0,1,2 preserved)
            # 0: CLS
            # 1: SEP
            # 2: PAD
            L = (L + 3)[:(max_len_L - 2)].tolist()
            L_ids = (
                    [0] +
                    L +
                    [1] +
                    [2] * (max_len_L - 2 - len(L))
            )
            L_mask = [1] * (len(L) + 2) + [0] * (max_len_L - 2 - len(L))
            input_ids_L[i] = torch.tensor(L_ids)
            attn_mask_L[i] = torch.tensor(L_mask)

            R = (R + 3)[:(max_len_R - 2)].tolist()
            R_ids = (
                    [0] +
                    R +
                    [1] +
                    [2] * (max_len_R - 2 - len(R))
            )
            R_mask = [1] * (len(R) + 2) + [0] * (max_len_R - 2 - len(R))
            input_ids_R[i] = torch.tensor(R_ids)
            attn_mask_R[i] = torch.tensor(R_mask)

            labels[i] = label
            cases.append(case)

        return {
            "input_ids_L": input_ids_L.long(),
            "attn_mask_L": attn_mask_L.bool(),
            "input_ids_R": input_ids_R.long(),
            "attn_mask_R": attn_mask_R.bool(),
            "label": labels,
            "case": cases
        }
    
class SeqAttnDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str,
        batch_size: int, num_workers: int,
        k_fold: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.k_fold = k_fold
        self.dataset = {}
    
    def setup(self, stage: Optional[str] = None):
        for split in ['train', 'valid', 'test']:
            file = os.path.join(self.data_dir, f"k{self.k_fold}", f"{split}_processed.pkl")
            assert os.path.exists(file)

            with open(file, "rb") as f:
                seq = pickle.load(f)
            
            self.dataset[split] = SeqAttnData(seq)
    
    def get_loader(self, split) -> DataLoader:
        return DataLoader(
            self.dataset[split], batch_size=self.batch_size,
            collate_fn=SeqAttnData.collate_fn, drop_last=False,
            num_workers=self.num_workers, pin_memory=True,
            shuffle=(split=="train")
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.get_loader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_loader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.get_loader("test")