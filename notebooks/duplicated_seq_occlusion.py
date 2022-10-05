import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, Dataset)

import pickle
import sys

from torch.utils.data.dataset import ConcatDataset

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
from src.module.model.TwoHeadLSTM import TwoHeadLSTM, elementwise_apply
from src.data.SeqData import SeqData

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, IntegratedGradients
from itertools import groupby

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

k_fold = 3
CKPT_PATH = "/home/jiashu/seq/selected_ckpt/565/ckpt/epoch00-AUROC0.75-acc0.00.ckpt"
model = TwoHeadLSTM()
state_dict = torch.load(CKPT_PATH)['state_dict']
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

datasets = []
for split in ["train", "valid", "test"]:
    DATA_PATH = f"../processed/k{k_fold}/{split}_processed,pkl"
    with open(DATA_PATH, "rb") as f:
        seq = pickle.load(f)
    datasets.append(SeqData(seq))

ds = ConcatDataset(datasets)


dataloader = DataLoader(
    ds, batch_size=1,
    collate_fn=SeqData.collate_fn, drop_last=False,
    num_workers=1, pin_memory=True
)

gesture_mapping = {}
with open(f"/home/jiashu/seq/processed/k{k_fold}/gesture_mapping.txt") as f:
    for line in f:
        gesture, gid = line.rstrip().split("->")
        gesture_mapping[int(gid)] = gesture


####################################################################################################
# Occlusion

records = {}

@torch.no_grad()
def occlusion(i, batch):
    # orig_prob = model(**batch).sigmoid()
    orig_logits = model(**batch)
    record = []
    for LR in ["L", "R"]:
        occluded_seqs = []
        starting = 0
        seq = batch[LR]["gesture"].data.cpu().numpy()
        for ele, groups in groupby(seq):
            groups = list(groups)
            if len(groups) >= 3:
                occluded_seq = seq.copy()
                # other
                occluded_seq[starting: starting + len(groups)] = -1
                occluded_seqs.append((occluded_seq, groups, starting))

            starting += len(groups)

        # dict of starting idx -> (groups, orig - cur logits)
        occlusion_scores = {}
        for occluded_seq, groups, starting in occluded_seqs:
            new_batch = copy.deepcopy(batch)
            new_batch[LR]["gesture"] = PackedSequence(
                torch.from_numpy(occluded_seq), batch_sizes=new_batch[LR]["gesture"].batch_sizes)
            # prob = model(**new_batch).sigmoid()
            logits = model(**new_batch)
            occlusion_scores[starting] = (
                [gesture_mapping[ele] for ele in groups],
                (orig_logits - logits).item()
            )

        record.append(occlusion_scores)
    record.append(batch['label'].item())
    record.append(batch['id'][0])
    records[i] = record


for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    occlusion(i, batch)

with open("duplicated_seq_occlusion.pl", "wb") as f:
    pickle.dump(records, f)