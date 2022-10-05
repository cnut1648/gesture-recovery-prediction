import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, Dataset)

import pickle
import sys

from torch.utils.data.dataset import ConcatDataset
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append("../")
import copy

import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import tqdm

# from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, AutoModel

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from src.module.BaselineModule import BaselineModule
from src.module.model.TwoHeadIMVLSTM import TwoHeadIMVLSTM
from src.data.SeqData import SeqData

from itertools import groupby

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

k_fold = 1
CKPT_PATH = "/home/jiashu/seq/selected_ckpt/3262/epoch02-AUROC0.20-acc0.00.ckpt"
model: TwoHeadIMVLSTM = BaselineModule.load_from_checkpoint(CKPT_PATH).model

model.eval()
model.zero_grad()

emb_L = model.embeddings_L
emb_R = model.embeddings_R

d = {}
for split in ["valid", "test"]:
    DATA_PATH = f"../processed/k{k_fold}/{split}_processed,pkl"
    with open(DATA_PATH, "rb") as f:
        seq = pickle.load(f)
    ds = SeqData(seq)

    dataloader = DataLoader(
        ds, batch_size=20,
        collate_fn=SeqData.collate_fn, drop_last=False,
        num_workers=1, pin_memory=True
    )

    for data in dataloader:
        logits = model(**data)
        logits = logits.detach().cpu().numpy()
        pred = logits > 0.011193
        label = data['label'].detach().cpu().numpy()
        print(f"logits = {logits}, label={label}")
        print(f"acc = {accuracy_score(pred, label)}, roc = {roc_auc_score(label, logits)}")

        d[split] = (logits, label)
        print()


gesture_mapping = {}
with open(f"/home/jiashu/seq/processed/k{k_fold}/gesture_mapping.txt") as f:
    for line in f:
        gesture, gid = line.rstrip().split("->")
        gesture_mapping[int(gid)] = gesture


####################################################################################################
# IMV Vis

# def emb(packed_sequence: PackedSequence):
#     embs = []
#     for gesture in packed_sequence.data.unbind():
#         # occlusion, use -1 for zero emb
#         if gesture.item() == -1:
#             embs.append(model.zero_emb)
#         else:
#             embs.append(model.embeddings_L(gesture))
#     embs = torch.stack(embs, dim=0)
#     return PackedSequence(embs, packed_sequence.batch_sizes)

# with torch.no_grad():
#     for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # id = batch['id'][0]
        
        # gesture_L = batch["L"]["gesture"]
        # emb_gesture_L: PackedSequence = emb(gesture_L)
        # pad_L, len_L = pad_packed_sequence(emb_gesture_L, batch_first=True)

        # gesture_R: PackedSequence = batch["R"]["gesture"]
        # emb_gesture_R: PackedSequence = emb(gesture_R)
        # pad_R, len_R = pad_packed_sequence(emb_gesture_R, batch_first=True)

        # gesture_L = [
        #     f"{gesture_mapping[g]}-{i}"
        #     for i, g in enumerate(gesture_L.data.numpy())
        # ]
        # gesture_R = [
        #     f"{gesture_mapping[g]}-{i}"
        #     for i, g in enumerate(gesture_R.data.numpy())
        # ]
        
        # fig, (ax1, ax2) = plt.subplots(
        #     1, 2,
        #     figsize=(8, 160))
        
        # # pad_L (bsz=1, T = 320, d = 166)
        # # output (bsz=1, hidden_dim = 367)
        # # alpha (bsz=1, T = 320, d = 166, 1)
        # # beta (bsz=1, d = 166, 1)
        # output, alpha, beta = model.lstm(pad_L)
        # # (T, d)
        # alpha = alpha.squeeze()
        # # (d, )
        # beta = beta.squeeze()
        
        # im = ax1.imshow(alpha.mean(1, keepdim=True))
        # ax1.set_xticks([0])
        # ax1.set_yticks(np.arange(len(gesture_L)))
        # ax1.set_xticklabels(["L"])
        # ax1.set_yticklabels(gesture_L)
        
        # output, alpha, beta = model.lstm(pad_R)
        # # (T, d)
        # alpha = alpha.squeeze()
        # # (d, )
        # beta = beta.squeeze()
        
        # im = ax2.imshow(alpha.mean(1, keepdim=True))
        # ax2.set_xticks([0])
        # ax2.set_yticks(np.arange(len(gesture_R)))
        # ax2.set_xticklabels(["R"])
        # ax2.set_yticklabels(gesture_R)
        
        # plt.colorbar(im, ax=ax2)
        # fig.savefig(f"IMV2/{i}-{id}.jpeg", bbox_inches='tight', pad_inches=1)

        # plt.close()