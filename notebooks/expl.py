import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, Dataset)
import copy

import pickle
import sys
sys.path.append("../")

import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import tqdm

# from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, AutoModel

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from src.module.BaselineModule import BaselineModule
from src.module.model.TwoHeadLSTM import TwoHeadLSTM, elementwise_apply
from src.data.SeqData import SeqData


from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, IntegratedGradients


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_fold = 3
CKPT_PATH = "/home/jiashu/seq/selected_ckpt/565/ckpt/epoch00-AUROC0.75-acc0.00.ckpt"
# TwoHeadLSTM
# model = BaselineModule.load_from_checkpoint(CKPT_PATH).model
model = TwoHeadLSTM()

# model = model.to(device)
# model.eval()
# model.zero_grad()

DATA_PATH = f"../processed/k{k_fold}/train_processed,pkl"
with open(DATA_PATH, "rb") as f:
    seq = pickle.load(f)
sd = SeqData(seq)
torch.backends.cudnn.enabled=False

dataloader = DataLoader(
        sd, batch_size=1,
        collate_fn=SeqData.collate_fn, drop_last=False,
        num_workers=1, pin_memory=True
)
batch = next(iter(dataloader))
def forward_lig(L_seq, R_seq, L_len, R_len):
    # eval:
    #   L_seq (1, s)
    #   L_int int = s
    # IG:
    #   L_seq (n_step, s)
    #   L_int int = s
    L = pack_padded_sequence(
        L_seq.T, L_len, enforce_sorted=False
    ).to(device)
    R = pack_padded_sequence(
        R_seq.T, R_len, enforce_sorted=False
    ).to(device)
    L = {"gesture": L}
    R = {"gesture": R}
    # (N=1, )
    logits = model(L, R)
    # pseudo-prob
    return logits.sigmoid()

####################################################################################################
# LAYER IG
# ig = LayerIntegratedGradients(forward_lig, [model.embeddings_L, model.embeddings_R])

# L_pack = batch["L"]["gesture"]
# R_pack = batch["R"]["gesture"]
# L_seq, L_len = pad_packed_sequence(L_pack)
# R_seq, R_len = pad_packed_sequence(R_pack)
# # (1, length)
# # L_len int = length
# L_seq = L_seq.view(1, -1)
# R_seq = R_seq.view(1, -1)
# attr_L, attr_R = ig.attribute(
#     inputs=(L_seq, R_seq),
#     additional_forward_args=(L_len, R_len),
#     baselines=(torch.zeros_like(L_seq), torch.zeros_like(R_seq)),
#     return_convergence_delta=False)

# print(attr_L.shape)

####################################################################################################
# Occlusion
from itertools import groupby
seq_L = batch["L"]["gesture"].data.cpu().numpy()
seq_R = batch["R"]["gesture"].data.cpu().numpy()

orig_logits = model(**batch)

occluded_seqs = []
starting = 0
for ele, groups in groupby(seq_L):
    groups = list(groups)
    if len(groups) >= 3:
        occluded_seq = seq_L.copy()
        # other
        occluded_seq[starting: starting + len(groups)] = 18
        occluded_seqs.append((occluded_seq, groups, starting))

    starting += len(groups)

occlusion_scores = {}
for occluded_seq, groups, starting in occluded_seqs:
    new_batch = copy.deepcopy(batch)
    new_batch["L"]["gesture"] = PackedSequence(torch.from_numpy(occluded_seq), batch_sizes=new_batch["L"]["gesture"].batch_sizes)
    logits = model(**new_batch)
    occlusion_scores[starting] = (
        groups,
        (orig_logits - logits).item()
    )

print(occlusion_scores)

####################################################################################################
# def forward_cond(L_emb, R_emb, L_len, R_len):
#     # baseline
#     if type(L_emb) is not PackedSequence:
#         L_emb = pack_padded_sequence(
#             L_emb.T, L_len, enforce_sorted=False
#         )
#         R_emb = pack_padded_sequence(
#             R_emb.T, R_len, enforce_sorted=False
#         )
#     logits = model(None, None, L_emb=L_emb, R_emb=R_emb)
#     # pseudo-prob
#     return logits.sigmoid()

# cond = LayerConductance(forward_cond, 
#         model.linear)
# batch = next(iter(dataloader))
# L_pack = batch["L"]["gesture"]
# R_pack = batch["R"]["gesture"]

# L_seq, L_len = pad_packed_sequence(L_pack)
# R_seq, R_len = pad_packed_sequence(R_pack)
# # (1, length)
#     # L_len int = length
# L_seq = L_seq.view(1, -1)
# R_seq = R_seq.view(1, -1)

# L_emb = elementwise_apply(model.embeddings_L, L_pack)
# R_emb = elementwise_apply(model.embeddings_R, R_pack)
 