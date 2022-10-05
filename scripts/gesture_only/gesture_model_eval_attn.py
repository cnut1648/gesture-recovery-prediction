"""
eval gesture model on new 10 cases
"""
from pathlib import Path
import os, sys


import pickle
import numpy as np
import torch
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from copy import deepcopy
import sys; sys.path.insert(0, "/home/jiashu/seq")
print(sys.path)

from src.module.BaselineModule import BaselineModule
from src.data.SeqAttnDataModule import SeqAttnData
from src.module.model.AttnModel import AttnModel

@torch.no_grad()
def get_auc_for_kfold(ckpt, kfold, gestures: dict) -> float:
    gesture_mapping = {}
    with open(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/gesture_mapping.txt") as f:
        for line in f:
            gesture, gid = line.rstrip().split("->")
            gesture_mapping[gesture] = int(gid)
    
    model: AttnModel = BaselineModule.load_from_checkpoint(ckpt).model
    model.eval()
    model.zero_grad()

    logits = []
    labels = []
    gestures_clone = deepcopy(gestures)

    for LR in ["L", "R"]:
        for case in gestures_clone:
            gestures_clone[case][LR]["gesture"] = np.array([
                gesture_mapping[value]
                        for value in gestures_clone[case][LR]['gesture']
            ])
    dataset = SeqAttnData(gestures_clone)
    dataloader = DataLoader(
        dataset, batch_size=len(dataset),
        collate_fn=SeqAttnData.collate_fn, drop_last=False,
        num_workers=10, pin_memory=True
    )
    with torch.no_grad():
        for batch in dataloader:
            logits = model(**batch)
            labels = batch["label"]

    auc = roc_auc_score(labels, logits)
    return auc

def calc_best_test_auc(ckpt_id):
    gesture_model_ckptids = [
        ckpt_id + i
        for i in range(4)
    ]

    model_ckpts = {}

    for k, id in enumerate(gesture_model_ckptids):
        root = Path("logs/save/") / str(id) / "ckpt"
        ckpts = os.listdir(root)
        ckpts = [
            ckpt
            for ckpt in ckpts
            if ckpt != "last.ckpt"
        ]
        print(id)
        for ckpt in ckpts:
            print("\t", ckpt)
        print()
        model_ckpts[k] = [
            root / ckpt
            for ckpt in ckpts
        ]
    all_choices = []
    for k0 in model_ckpts[0]:
        for k1 in model_ckpts[1]:
            for k2 in model_ckpts[2]:
                for k3 in model_ckpts[3]:
                    all_choices.append([k0, k1, k2, k3])


    with open("/home/jiashu/seq/processed/train30_test10/new_10_cases.pkl", "rb") as f:
        gestures = pickle.load(f)
    best = -1
    with open(f"logs/x/{ckpt_id}", "w") as f:
        for choice in all_choices:
            rets = []
            for kfold, ckpt in enumerate(choice):
                auc = get_auc_for_kfold(ckpt, kfold, gestures)
                rets.append(auc)
            rets = np.array(rets)
            rets = np.where(rets > 0.5, rets, 1-rets)
            print(choice)

            f.write(str(choice))
            f.write("\n")

            if rets.mean() > best:
                best = rets.mean()

            f.write(f"\t {rets}, {rets.mean()} ({rets.std(ddof=1)})\t")
            f.write(f"current best is {best}")
            f.write("\n")

# for id in [4410, 4414, 4402, 4394, ]:
for id in [14246, 14275, 14260, 14307, 14300]:
    calc_best_test_auc(id)