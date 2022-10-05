import pickle
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import (DataLoader, Dataset)

class SeqData(Dataset):
    def __init__(self, seq: dict):
        self.seq = seq
        self.caseids = list(seq)

    def __len__(self):
        return len(self.caseids)


    def __getitem__(self, idx: int) -> dict:
        """
        return dict of this caseid
            - label -> binary 0 or 1
            - L -> data
            - R -> data
            - score -> dict use for eval model, only exist when is test = True (already pruned in DataModule)
        data = {
            nerve sparing: one ternary int 0 1 2
            gesture: main seq
            arm: seq, whether it uses three arms, ternary 0 1 2
            note: seq, str, ???
        }
        """
        caseid = self.caseids[idx]
        seq = self.seq[caseid]
        seq['id'] = caseid
        return seq

    @staticmethod
    def collate_fn(batches: List[dict]) -> dict:
        """
        ret L, R, label (N, ), scores (None if is_test = False; if is test dict of name -> (N, ))
        L or R dict
           -  gesture: main seq
           -  arm: main seq
           -  note: seq ????
           -  nerve sparing: (N, ) ternary int 0 1 2
        """
        labels: List[int] = []
        ids: List[str] = []
        Ls = defaultdict(list)
        Rs = defaultdict(list)
        
        is_test = True if "scores" in batches[0] else False
        scores = defaultdict(list) if is_test else None
        for batch in batches:
            labels.append(batch["label"])
            ids.append(batch['id'])
            if is_test:
                for name, score in batch["scores"].items():
                    scores[name].append(score)
            for LR, LRs in zip(["L", "R"], [Ls, Rs]):
                for k, v in batch[LR].items():
                    LRs[k].append(v)
        # TODO
        # rm note for now
        for LRs in [Ls, Rs]:
            LRs["gesture"] = pack_sequence(
                [torch.tensor(arr) for arr in LRs["gesture"]], enforce_sorted=False
            )
            # LRs["g"] = (
            #     [torch.tensor(arr) for arr in LRs["gesture"]]
            # )
            # LRs["arm"] = pack_sequence(
            #     [torch.tensor(arr) for arr in LRs["arm"]], enforce_sorted=False
            # )
            # LRs["nerve sparing"] = torch.tensor(
            #     LRs["nerve sparing"]
            # )
                
            

        if is_test:
            scores = {
                name: torch.tensor(score)
                for name, score in scores.items()
            }
        return {
            "L": Ls, 
            "R": Rs,
            "label": torch.tensor(labels).long(),
            "scores": scores,
            "id": ids
        }
