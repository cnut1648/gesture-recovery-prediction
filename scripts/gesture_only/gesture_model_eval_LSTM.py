"""
eval gesture model on new 10 cases
"""

import pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_sequence
from copy import deepcopy
import sys; sys.path.insert(0, "/home/jiashu/seq")
print(sys.path)

from src.module.BaselineModule import BaselineModule
from src.module.model.TwoHeadIMVLSTM import TwoHeadIMVLSTM

gesture_model_ckpts = [
    "/home/jiashu/seq/selected_ckpt/3667/epoch06-AUROC0.62-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3669/epoch10-AUROC0.71-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3736/epoch03-AUROC0.64-acc0.50.ckpt",
    "/home/jiashu/seq/selected_ckpt/3681/epoch09-AUROC0.74-acc0.50.ckpt",
]


@torch.no_grad()
def get_auc_for_kfold(kfold, gestures: dict) -> float:
    gesture_mapping = {}
    with open(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/gesture_mapping.txt") as f:
        for line in f:
            gesture, gid = line.rstrip().split("->")
            gesture_mapping[gesture] = int(gid)
    
    model: TwoHeadIMVLSTM = BaselineModule.load_from_checkpoint(
        gesture_model_ckpts[kfold]).model
    model.eval()
    model.zero_grad()

    logits = []
    labels = []
    gestures_clone = deepcopy(gestures)
    with torch.no_grad():
        for case in gestures_clone:
            labels.append(gestures_clone[case]['label'])
            for LR in ["L", "R"]:
                gestures_clone[case][LR]['gesture'] = pack_sequence(torch.tensor([[
                    gesture_mapping[value]
                    for value in gestures_clone[case][LR]['gesture']
                ]]))
            logit = model(**gestures_clone[case]).item()
            logits.append(logit)
    auc = roc_auc_score(labels, logits)
    return auc

with open("/home/jiashu/seq/processed/train30_test10/new_10_cases.pkl", "rb") as f:
    gestures = pickle.load(f)
rets = []
for kfold in range(4):
    auc = get_auc_for_kfold(kfold, gestures)
    rets.append(auc)
rets = np.array(rets)
rets = np.where(rets > 0.5, rets, 1-rets)
print(rets, rets.mean(), rets.std(ddof=1))
