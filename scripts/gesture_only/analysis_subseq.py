
from pathlib import Path
import pickle
import sys; sys.path.insert(0, "/home/jiashu/seq")
import torch
import numpy as np


from src.module.BaselineModule import BaselineModule
from src.data.SeqAttnDataModule import SeqAttnData
from src.module.model.AttnModel import AttnModel

patterns = [(48, [2, 2, 2, 2, 2]),
    (41, [13, 13, 13,  2, 13]),
    (30, [6, 8, 2, 2, 2]), 
    (26, [13, 13,  2,  2 , 2]),
    (25, [8, 2, 2, 2, 2]),
    (24, [15,  2,  2,  2,  2]),
    (24, [13, 13, 13,  2,  2]),
    (19, [10, 15,  2,  2,  2]),
    (19, [ 2, 13,  2,  2,  2]),
    (16, [15, 10,  2,  2, 13]),
    (16, [15, 10,  2,  2,  2]),
    (16, [13,  2,  2,  2,  2]),
    (15, [13, 13,  2, 13,  2]),
    (13, [13, 15,  2,  2,  2]),
    (13, [10, 15,  2,  2, 13])]

if __name__ == "__main__":
    ckpts = [
        "logs/save/4502/ckpt/epoch75-AUROC0.85-acc0.60.ckpt",
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
            pattern2attn_L = {
                tuple(pat): {
                    "label 0": [],
                    "label 1": [],
                }
                for _, pat in patterns
            }
            pattern2attn_R = {
                tuple(pat): {
                    "label 0": [],
                    "label 1": [],
                }
                for _, pat in patterns
            }
            for case, d in layer_attns.items():
                score_L, score_R = d['attn_score']
                gesture_L, gesture_R = d["orig_gesture"]

                batch = [(
                    case, gesture_L, gesture_R, d['label']
                )]

                batch = SeqAttnData.collate_fn(batch)
                key = f"label {int(d['label'])}"

                for pattern in pattern2attn_L:
                    pattern = list(pattern)
                    n = len(pattern)
                    for i in range(len(gesture_L)):
                        if gesture_L[i:i+n].tolist() == pattern:
                            occlusion_idx = np.ones((1, len(gesture_L)), dtype=int)
                            occlusion_idx[:, i: i + n] = 0
                            occlusion_idx = torch.from_numpy(occlusion_idx).bool()
                            kwargs = {
                                f"occlusion_idx_L": occlusion_idx
                            }

                            new_logits = model(**batch, **kwargs)
                            delta = (d['orig_logits'] - new_logits).item()

                            pattern2attn_L[tuple(pattern)][key].append((case, delta))

                for pattern in pattern2attn_R:
                    pattern = list(pattern)
                    n = len(pattern)
                    for i in range(len(gesture_R)):
                        if gesture_R[i:i+n].tolist() == pattern:
                            occlusion_idx = np.ones((1, len(gesture_R)), dtype=int)
                            occlusion_idx[:, i: i + n] = 0
                            occlusion_idx = torch.from_numpy(occlusion_idx).bool()
                            kwargs = {
                                f"occlusion_idx_R": occlusion_idx
                            }

                            new_logits = model(**batch, **kwargs)
                            delta = (d['orig_logits'] - new_logits).item()

                            pattern2attn_R[tuple(pattern)][key].append((case, delta))
            
            with open(f"{layer}_Lpattern[k=1].pkl", "wb") as f:
                pickle.dump(pattern2attn_L, f)
            with open(f"{layer}_Rpattern[k=1].pkl", "wb") as f:
                pickle.dump(pattern2attn_R, f)