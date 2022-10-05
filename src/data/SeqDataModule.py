import pickle, os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from src.utils.utils import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import (DataLoader, Dataset)
from src.data.SeqData import SeqData

log = get_logger(__name__)
class SeqDataModule(pl.LightningDataModule):
    def __init__(self,
        data_dir: str,
        batch_size: int, num_workers: int,
        num_distinct_gestures: int,
        # if use all data and split
        # eg. 10:10 i.e. 10% & 10%
        # not in use if k_fold
        valid_test: str,
        # use k-fold
        k_fold: Optional[int] = None,
        # need to creat test
        has_test: bool = True,
        **kwargs
     ):
        super().__init__()
        self.valid_test = valid_test
        self.has_test = has_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_distinct_gestures = num_distinct_gestures
        self.data_dir = data_dir
        self.k_fold = k_fold
        self.dataset = {}
        # convert gesture (eg s for spread) to int
        self.converter = LabelEncoder()

    def load_seq_data(self, caseids: List[str], is_test: bool) -> Dict[str, dict]:
        """
        load seq data for caseid
        # TODO
        ignore notes for now, make notes to score
        convert gesture to int
        """
        ret_dict = {}
        for caseid in caseids:
            seq = self.seq[caseid]
            ret = {}
            # 0 or 1
            ret["label"] = seq["label"]

            if is_test:
                assert "scores" in seq
                # dict
                ret["scores"] = seq["scores"]

            for LR in ["L", "R"]:
                ret[LR] = {
                    "nerve sparing": seq[LR]["Nerve Sparing"],
                    # list of 0/1/2
                    "arm": seq[LR]["Arm"],
                    # list of int
                    # "gesture": self.converter.transform(list(map(lambda s: s.strip(), seq[LR]["Gesture"])))
                    "gesture": self.converter.transform(list(seq[LR]["Gesture"]))
                }
            
            ret_dict[caseid] = ret
        return ret_dict


    def setup(self, stage: Optional[str] = None):
        if self.k_fold is not None:
            assert type(self.k_fold) is int
            k_dir = os.path.join(self.data_dir, f"k{self.k_fold}")
            assert os.path.exists(k_dir)
            log.info(f"getting kfold from {k_dir}")

            with open(os.path.join(k_dir, "train.pkl"), "rb") as f:
                train_seq = pickle.load(f)
            with open(os.path.join(k_dir, "valid.pkl"), "rb") as f:
                valid_seq = pickle.load(f)
            
            seq = {
                **train_seq,
                **valid_seq,
            }
            if self.has_test:
                with open(os.path.join(k_dir, "test.pkl"), "rb") as f:
                    test_seq = pickle.load(f)
                seq.update(test_seq)

            all_gestures = [
                gesture
                for value_dict in seq.values()
                for LR in ["L", "R"]
                for gesture in value_dict[LR]["Gesture"]
            ]

            assert len(set(all_gestures)) == self.num_distinct_gestures
            self.converter.fit_transform(all_gestures)
            gesture_mapping = {
                g : i
                for i, g in enumerate(self.converter.classes_)
            }
            with open(os.path.join(k_dir, "gesture_mapping.txt"), "w") as f:
                for g, i in gesture_mapping.items():
                    f.write(f"{g}->{i}\n")

            phases = ["train", "valid"]
            seqs = [train_seq, valid_seq]
            if self.has_test:
                phases.append("test")
                seqs.append(test_seq)

            for phase, seq in zip(phases, seqs):
                # load_seq_data need self.seq
                self.seq = seq
                seq_data = self.load_seq_data(list(seq), is_test=False)
                self.dataset[phase] = SeqData(seq=seq_data)
                with open(os.path.join(k_dir, f"{phase}_processed.pkl"), "wb") as f:
                    pickle.dump(seq_data, f)

        else:
            with open(os.path.join(self.data_dir, "seq.pkl"), "rb") as f:
                # case id -> dict
                # dict keys:
                # - label 0 or 1
                # - L data
                # - R data
                self.seq: dict = pickle.load(f)

            # generate
            all_gestures = [
                gesture
                for value_dict in self.seq.values()
                for LR in ["L", "R"]
                for gesture in value_dict[LR]["Gesture"]
            ]
            assert len(set(all_gestures)) == self.num_distinct_gestures
            self.converter.fit_transform(all_gestures)

            all_caseids = list(self.seq.keys())
            caseids_with_scores = [
                caseid
                for caseid in all_caseids
                if "scores" in self.seq[caseid]
            ]
            caseids_without_scores = [
                caseid
                for caseid in all_caseids
                if "scores" not in self.seq[caseid]
            ]

            # make sure test caseid always has score
            valid_ratio, test_ratio = map(lambda ratio: int(ratio) / 100, self.valid_test.split(":"))
            num_test = min(round(test_ratio * len(all_caseids)), len(caseids_with_scores))
            test_caseids = caseids_with_scores[:num_test]
            new_test_ratio = num_test / len(all_caseids)

            valid_ratio = valid_ratio + test_ratio - new_test_ratio

            train_caseids, valid_caseids = train_test_split(
                caseids_without_scores + caseids_with_scores[num_test:],
                test_size=valid_ratio)

            assert len(train_caseids) + len(valid_caseids) + len(test_caseids) == len(all_caseids)

            self.dataset["train"] = SeqData(seq=self.load_seq_data(train_caseids, is_test=False))
            self.dataset["valid"] = SeqData(seq=self.load_seq_data(valid_caseids, is_test=False))
            self.dataset["test"] = SeqData(seq=self.load_seq_data(test_caseids, is_test=True))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"], batch_size=self.batch_size,
            collate_fn=SeqData.collate_fn, drop_last=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["valid"], batch_size=self.batch_size,
            collate_fn=SeqData.collate_fn, drop_last=False,
            num_workers=self.num_workers, pin_memory=True
        )
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"], batch_size=self.batch_size,
            collate_fn=SeqData.collate_fn, drop_last=False,
            num_workers=self.num_workers, pin_memory=True
        )