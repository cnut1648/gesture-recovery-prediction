"""
after extract_attn_scores.py
"""

from pathlib import Path
import os, sys


import pickle
import numpy as np
import torch
from pathlib import Path
import os
from itertools import groupby
import sys; sys.path.insert(0, "/home/jiashu/seq")
print(sys.path)
from extract_attn_scores import get_all_gesture_attn_dataloader

from src.module.BaselineModule import BaselineModule
from src.data.SeqAttnDataModule import SeqAttnData
from src.module.model.AttnModel import AttnModel

@torch.no_grad()
def occlusion(model, batch, orig_logits, score, gesture, LR: str):
    above_threshold = score > score.mean()
    starting = 0
    occl_score = np.zeros(len(gesture), dtype=float)

    for ele, groups in groupby(above_threshold):
        groups = list(groups)
        # only consider subseq w/ len >= 3
        if ele == True and len(groups) >= 3:
            occlusion_idx = np.ones((1, len(gesture)), dtype=int)
            occlusion_idx[:, starting: starting + len(groups)] = 0
            occlusion_idx = torch.from_numpy(occlusion_idx).bool()
            kwargs = {
                f"occlusion_idx_{LR}": occlusion_idx
            }

            new_logits = model(**batch, **kwargs)

            delta = (orig_logits - new_logits).item()
            occl_score[starting: starting + len(groups)] = delta

        starting += len(groups)
    return occl_score


if __name__ == "__main__":
    ckpts = [
        "logs/save/4502/ckpt/epoch75-AUROC0.85-acc0.60.ckpt",
        "logs/save/4503/ckpt/epoch33-AUROC0.94-acc0.60.ckpt",
        "logs/save/4504/ckpt/epoch28-AUROC0.80-acc0.50.ckpt",
        "logs/save/4505/ckpt/epoch95-AUROC0.88-acc0.50.ckpt",
    ]
    artifact_dir = Path("artifact/transformer/")
    data_dir = Path("processed/train30_test10/")

    for k, ckpt in enumerate(ckpts):
        model: AttnModel = BaselineModule.load_from_checkpoint(ckpt).model
        model.eval()
        model.zero_grad()

        with open(artifact_dir / f"k{k}.pkl", "rb") as f:
            # {
            #  layer_mode => {
            #     case => 
            #      Key[attn_scores, label, split, orig_logits, orig_gesture]
            #      Value[(L, R), 1/0, str, float, (L, R)]
            #     # to add: Key[occl_scores] => (L, R)
            # }}
            attns = pickle.load(f)

        for layer, layer_attns in attns.items():
            for case, d in layer_attns.items():
                score_L, score_R = d['attn_score']
                gesture_L, gesture_R = d["orig_gesture"]

                batch = [(
                    case, gesture_L, gesture_R, d['label']
                )]

                batch = SeqAttnData.collate_fn(batch)

                attns[layer][case]["occl_scores"] = (
                    occlusion(model, batch, d["orig_logits"], score_L, gesture_L, LR="L"),
                    occlusion(model, batch, d["orig_logits"], score_R, gesture_R, LR="R"),
                )


        with open(artifact_dir / f"k{k}.pkl", "wb") as f:
            pickle.dump(attns, f)