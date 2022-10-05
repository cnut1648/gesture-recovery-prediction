import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, Dataset)

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
device = torch.device("cpu")
print(device)

k_fold = 3
CKPT_PATH = "/home/jiashu/seq/selected_ckpt/565/ckpt/epoch00-AUROC0.75-acc0.00.ckpt"
model = TwoHeadLSTM()
state_dict = torch.load(CKPT_PATH, map_location=device)['state_dict']
new_dict = {}
for k, w in state_dict.items():
    if k == "model.embeddings.weight":
        new_dict['embeddings_L.weight'] = w
        new_dict['embeddings_R.weight'] = w
    else:
        k = k.replace("model.", "")
        new_dict[k] = w
model.load_state_dict(new_dict)

# model = model.to(device)
model.eval()
model.zero_grad()

DATA_PATH = f"../processed/k{k_fold}/train_processed,pkl"
with open(DATA_PATH, "rb") as f:
    seq = pickle.load(f)
sd = SeqData(seq)

dataloader = DataLoader(
    sd, batch_size=1,
    collate_fn=SeqData.collate_fn, drop_last=False,
    num_workers=1, pin_memory=True
)

gesture_mapping = {}
with open(f"/home/jiashu/seq/processed/k{k_fold}/gesture_mapping.txt") as f:
    for line in f:
        gesture, gid = line.rstrip().split("->")
        gesture_mapping[int(gid)] = gesture

batch = next(iter(dataloader))


def forward_lig(L_seq, R_seq, L_len, R_len, target):
    # eval:
    #   L_seq (1, s)
    #   L_int int = s
    # IG:
    #   L_seq (n_step, s)
    #   L_int int = s
    L = pack_padded_sequence(
        L_seq.T, L_len, enforce_sorted=False
    # ).to(device)
    )
    R = pack_padded_sequence(
        R_seq.T, R_len, enforce_sorted=False
    # ).to(device)
    )
    L = {"gesture": L}
    R = {"gesture": R}
    # (N=1, )
    logits = model(L, R)
    # pseudo-prob
    prob = logits.sigmoid()
    return prob
    # if target == 0:
    #     return 1 - prob
    # return prob


def summarize(attr):
    # attr (length, hidden)
    sum = attr.sum(dim=-1)
    return sum / torch.norm(sum)


def vis(attr, prob, pred, label, raw_input):
    score_vis = viz.VisualizationDataRecord(
        word_attributions=attr,
        # this sent pred outcome
        pred_prob=prob,
        pred_class=pred,

        true_class=label,

        # `target`, which class attr based on
        attr_class=label,
        # overall attr score
        # magnitude shows strengh of importance
        # pos: 正面影响pred；neg： 负面
        attr_score=attr.sum(),

        # list of tokens, same shape as summary (s,)
        raw_input=raw_input,
        convergence_score=None)
    return score_vis


####################################################################################################
# LAYER IG
lig = LayerIntegratedGradients(forward_lig,
                               [model.embeddings_L, model.embeddings_R])

records = {}
torch.backends.cudnn.enabled=False

def lig_attr(i, batch):
    L_pack = batch["L"]["gesture"]
    R_pack = batch["R"]["gesture"]
    L_seq, L_len = pad_packed_sequence(L_pack)
    R_seq, R_len = pad_packed_sequence(R_pack)
    # (1, length)
    # L_len int = length
    L_seq = L_seq.view(1, -1)
    R_seq = R_seq.view(1, -1)

    attr_L, attr_R = lig.attribute(
        inputs=(L_seq, R_seq),
        additional_forward_args=(L_len, R_len, batch['label'].item()),
        baselines=(torch.zeros_like(L_seq), torch.zeros_like(R_seq)),
        return_convergence_delta=False)

    attr_L, attr_R = map(summarize, [attr_L, attr_R])

    prob = forward_lig(L_seq, R_seq, L_len, R_len, batch['label'].item())
    pred = prob >= (0.5 - 0.0020)

    records[i] = [
        vis(
            attr_L, prob.item(), pred.item(), batch["label"].item(),
            raw_input=[
                gesture_mapping[gid]
                for gid in L_pack.data.cpu().numpy()
            ]
        ),
        vis(
            attr_R, prob.item(), pred.item(), batch["label"].item(),
            raw_input=[
                gesture_mapping[gid]
                for gid in R_pack.data.cpu().numpy()
            ]
        )
    ]


for i, batch in enumerate(dataloader):
    lig_attr(i, batch)

with open("expl_no_1_minus.pl", "wb") as f:
    pickle.dump(records, f)
