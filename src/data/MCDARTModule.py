"""
using DART scores for Loc.xlsx
Monte Carlo
March 13 2022
"""

import os, random
import pickle
from typing import Optional
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SeqAttnData(Dataset):
    def __init__(self, seq: dict, data: dict, column_name: str):
        self.seq = seq
        self.data = data
        self.column_name = column_name
        self.idx2caseid = {
            i: case
            for i, case in enumerate(self.seq)
        }

    def __len__(self):
        return len(self.idx2caseid)

    def __getitem__(self, idx):
        case = self.idx2caseid[idx]
        data = self.seq[case]
        label = self.data[case][self.column_name]
        return case, data['L']['gesture'], data['R']['gesture'], label

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
    
class MCDARTModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str,
        column_name, 
        batch_size: int, num_workers: int,
        k_fold: int, 
        # if true, use below vs above median as target, not regression but classification
        target_as_clf: bool = False,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.column_name = column_name
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dataset = {}
        self.k_fold = k_fold
        self.target_as_clf = target_as_clf
    
    def setup(self, stage: Optional[str] = None):
        """
        load data
        """
        seq = {}
        for split in ['train', 'valid', 'test']:
            # file = os.path.join(self.data_dir, f"k0", f"{split}_processed.pkl")
            file = os.path.join(self.data_dir, f"k0", f"{split}_combined.pkl")
            assert os.path.exists(file)

            with open(file, "rb") as f:
                seq = {**seq, **pickle.load(f)}
            
        assert len(seq) == 50

        """
        load target
        """
        df = pd.read_excel("/home/jiashu/seq/raw_data/DART scores for Loc.xlsx", engine="openpyxl")
        data = {}
        for i in range(0, len(df), 2):
            name = tuple([df.iloc[i]["Video"], df.iloc[i+1]["Video"]])
            
            ave_th_2_sides = tuple([df.iloc[i]["ave_th_2_sides"], df.iloc[i+1]["ave_th_2_sides"]])
            ave_tr_2_sides = tuple([df.iloc[i]["ave_tr_2_sides"], df.iloc[i+1]["ave_tr_2_sides"]])
            ave_e_2_sides = tuple([df.iloc[i]["ave_e_2_sides"], df.iloc[i+1]["ave_e_2_sides"]])

            print(name, ave_th_2_sides, ave_tr_2_sides, ave_e_2_sides)
            
            data[name[0][:-2]] = {
                "ave_th_2_sides": ave_th_2_sides[0],
                "ave_tr_2_sides": ave_tr_2_sides[0],
                "ave_e_2_sides": ave_e_2_sides[0]
            }
        assert len(data) == 45

        if self.target_as_clf:
            for target in ["ave_th_2_sides", "ave_tr_2_sides", "ave_e_2_sides"]:
                all_targets = [
                    d[target]
                    for d in data.values()
                ]
                m = np.median(all_targets)
                for case in data:
                    if data[case][target] > m:
                        data[case][target] = 1
                    else:
                        data[case][target] = 0
        seq = {
            k:v for k, v in seq.items()
            if k in data
        }
        assert len(seq) == 40

        keys = list(seq.keys())
        random.seed(self.k_fold)
        random.shuffle(keys)
        traink, testk = keys[:30], keys[30:]
        train, test = {}, {}
        for k, v in seq.items():
            if k in traink:
                train = {**train, k:v}
            elif k in testk:
                test = {**test, k:v}
        self.dataset["train"] = SeqAttnData(train, data, self.column_name)
        self.dataset["valid"] = SeqAttnData(test, data, self.column_name)
    
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