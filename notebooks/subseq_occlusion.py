from collections import defaultdict
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

from numpy.lib.stride_tricks import sliding_window_view
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

@torch.no_grad()
def occlusion(i, batch, window_size):
    orig_logits = model(**batch)
    for LR in ["L", "R"]:
        record = []
        seq = batch[LR]["gesture"].data.cpu().numpy()
        indices = np.arange(seq.size)
        # each row is sliding window of size window_size
        # row is indices
        windows = sliding_window_view(indices, window_size)
        for window in windows:
            new_batch = copy.deepcopy(batch)
            occluded_seq = seq.copy()
            subseq = occluded_seq[window]
            occluded_seq[window] = -1
            new_batch[LR]["gesture"] = PackedSequence(
                torch.from_numpy(occluded_seq), batch_sizes=new_batch[LR]["gesture"].batch_sizes)

            logits = model(**new_batch)
            record.append((
                window,
                [gesture_mapping[ele] for ele in subseq],
                (orig_logits - logits).item()
            ))
        records[i].append(record)
    records[i].append(batch['label'].item())
    records[i].append(batch['id'][0])

for window_size in [3, 5, 10]:
    records = defaultdict(list)

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        occlusion(i, batch, window_size)

    with open(f"subseq_occlusion-{window_size}.pl", "wb") as f:
        pickle.dump(records, f)