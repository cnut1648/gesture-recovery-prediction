from argparse import ArgumentParser
import os
from pathlib import Path
import torch
from torch.utils.data import (DataLoader, Dataset)

import pickle
import sys

from torch.utils.data.dataset import ConcatDataset

sys.path.append("../")
from src.module.BaselineModule import BaselineModule
from src.module.model.TwoHeadIMVLSTM import TwoHeadIMVLSTM
import copy

import torch
from torch.nn.utils.rnn import PackedSequence
import tqdm

# from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, AutoModel

from src.data.SeqData import SeqData

from itertools import groupby

def parse():
    parser = ArgumentParser()
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

OUTPUT_DIR = Path(f"/home/jiashu/seq/artifact/attn_occl/{ckpt_id}")
if not OUTPUT_DIR.exists():
    os.makedirs(OUTPUT_DIR)

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
# for split in ["train", "valid"]:
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
ATTN_DIR = Path(f"../artifact/IMV/{ckpt_id}")
attns = {}
for p in os.listdir(ATTN_DIR):
    if p.endswith(".pkl"):
        pkl = ATTN_DIR / p
        with open(pkl, "rb") as f:
            attn = pickle.load(f)
        attns[p] = attn

####################################################################################################
# Occlusion

# cid -> L, R, label
records = {}

@torch.no_grad()
def occlusion(i, batch):
    orig_logits = model(**batch)
    id = batch['id'][0] + ".pkl"
    record = []
    for LR in ["L", "R"]:
        occluded_seqs = []
        starting = 0
        attn = attns[id][LR].view(-1).numpy()
        above_threshold = attn > attn.mean()
        seq = batch[LR]["gesture"].data.cpu().numpy()
        for ele, groups in groupby(above_threshold):
            groups = list(groups)
            if ele == True and len(groups) >= 3:
                occluded_seq = seq.copy()
                # other
                occluded_seq[starting: starting + len(groups)] = -1
                occluded_seqs.append((occluded_seq, seq[starting: starting + len(groups)], starting))

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
    records[batch['id'][0]] = record


for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    occlusion(i, batch)

with open(OUTPUT_DIR / "attn_occlusion.pl", "wb") as f:
    pickle.dump(records, f)