from argparse import ArgumentParser
from collections import defaultdict
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset)

import pickle
import sys

from torch.utils.data.dataset import ConcatDataset
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.append("../")
from src.module.BaselineModule import BaselineModule
from src.module.model.TwoHeadIMVLSTM import TwoHeadIMVLSTM
import copy

import torch
from torch.nn.utils.rnn import PackedSequence
import tqdm

from src.data.SeqData import SeqData

from itertools import groupby


def parse():
    parser = ArgumentParser()
    parser.add_argument("--coarse", default=False, action='store_true')
    parser.add_argument("--k", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--ckpt_id", required=True)
    return parser.parse_args()

arg = parse()
print(arg)

k_fold = arg.k
CKPT_PATH = arg.ckpt_path
assert os.path.exists(CKPT_PATH), f"{CKPT_PATH} not exist"
ckpt_id = arg.ckpt_id

COARSE = arg.coarse
coarse_str = "-coarse" if COARSE else ""
OUTPUT_DIR = Path(f"/home/jiashu/seq/artifact/attn_occl/{ckpt_id}{coarse_str}")
if not OUTPUT_DIR.exists():
    os.makedirs(OUTPUT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model: TwoHeadIMVLSTM = BaselineModule.load_from_checkpoint(CKPT_PATH).model

model.eval()
model.zero_grad()

#######
# dataset
gesture_mapping = {}
datasets = []
with open(f"/home/jiashu/seq/processed/train30_test10/k{k_fold}/gesture_mapping.txt") as f:
    for line in f:
        gesture, gid = line.rstrip().split("->")
        gesture_mapping[int(gid)] = gesture
for split in ["valid"]:
    DATA_PATH = f"../processed/train30_test10/k{k_fold}/{split}_processed.pkl"
    with open(DATA_PATH, "rb") as f:
        seq = pickle.load(f)
    datasets.append(SeqData(seq))

ds = ConcatDataset(datasets)


dataloader = DataLoader(
    ds, batch_size=1,
    collate_fn=SeqData.collate_fn, drop_last=False,
    num_workers=1, pin_memory=True
)

########
# attn scores (saved from )
with open(f"../artifact/attn_occl/{ckpt_id}/attn_occlusion.pl", "rb") as f:
    attns = pickle.load(f)

####################################################################################################
# Occlusion

@torch.no_grad()
def vis(i, batch):
    id = batch['id'][0]
    L, R, label  = attns[id]
    to_save = [label]

    fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(8, 160))

    # L
    gesture_L = batch['L']['gesture'].data
    gesture_L = [
            f"{gesture_mapping[g]}-{i}"
            for i, g in enumerate(gesture_L.data.numpy())
        ]

    startings = sorted(L.keys())
    if COARSE:
        scores = np.zeros_like(gesture_L, dtype=int)
        for start in startings:
            subseq, delta = L[start]
            if float(delta) > 0:
                score = 1
            elif float(delta) < 0: score = -1
            else: score = 0
            scores[start:start+len(subseq)] = score
        cmap = ListedColormap(['red', 'white', 'blue'])
        bounds=[-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)
        args = dict(
            cmap=cmap, norm=norm, 
            interpolation='nearest', origin='lower'
        )
    else:
        scores = np.zeros_like(gesture_L, dtype=float)
        for start in startings:
            subseq, delta = L[start]
            scores[start:start+len(subseq)] = delta
        args = {}
        # (T, 1)
    scores = scores[:, None]
    im = ax1.imshow(scores, **args)
    ax1.set_xticks([0])
    ax1.set_yticks(np.arange(len(gesture_L)))
    ax1.set_xticklabels(["L"])
    ax1.set_yticklabels(gesture_L)

    # Left
    to_save.append({
        "gesture": gesture_L,
        "score": scores
    })

    # R
    gesture_R = batch['R']['gesture'].data
    gesture_R = [
        f"{gesture_mapping[g]}-{i}"
        for i, g in enumerate(gesture_R.data.numpy())
    ]
    startings = sorted(R.keys())
    if COARSE:
        scores = np.zeros_like(gesture_R, dtype=float)
        for start in startings:
            subseq, delta = R[start]
            if float(delta) > 0:
                score = 1
            else: score = -1
            scores[start:start+len(subseq)] = score
    else:
        scores = np.zeros_like(gesture_R, dtype=float)
        for start in startings:
            subseq, delta = R[start]
            scores[start:start+len(subseq)] = delta
    # (T, 1)
    scores = scores[:, None]
    im = ax2.imshow(scores, **args)
    ax2.set_xticks([0])
    ax2.set_yticks(np.arange(len(gesture_R)))
    ax2.set_xticklabels(["R"])
    ax2.set_yticklabels(gesture_R)

    to_save.append({
        "gesture": gesture_R,
        "score": scores
    })

    plt.colorbar(im, ax=ax2)
    fig.savefig(f"{OUTPUT_DIR}/{id}.jpeg", bbox_inches='tight', pad_inches=1)

    plt.title(f"{id} [label = {label}]")

    plt.close()
    
    # save original scores (without binary)
    with open(f"{OUTPUT_DIR}/{id}.pkl", "wb") as f:
            pickle.dump(to_save, f)

for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    vis(i, batch)