"""
Generate clinic + gesture emb
NOTE run cp processed/train30_test10/k0/train_processed.pkl processed/train30_test10/k0/train_combined.pkl first
"""

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
import pandas as pd
from pathlib import Path
import pickle
import sys

from src.data.SeqAttnDataModule import SeqAttnData
sys.path.append("/home/jiashu/seq")
from pytorch_tabular.feature_extractor import DeepFeatureExtractor

from pytorch_tabular import TabularModel
import tqdm
from src.data.SeqData import SeqData

from src.module.BaselineModule import BaselineModule

data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

# SKIP_GESTURE = True
SKIP_GESTURE = False
GESTURE_IS_ATTN = True
# SKIP_TABULAR = True
SKIP_TABULAR = False

# if False, gen emb for valid + train
# if True, gen emb for test (i.e. from new_10_cases)
# GENERATE_TEST = True
GENERATE_TEST = False
splits = ["test"] if GENERATE_TEST else ["train", "valid"]

###############
# data
###############

if not GENERATE_TEST:
    df = pd.read_excel(
        data_dir / "Dissection Gesture_Clinical Data.xlsx",
        engine='openpyxl'
    )
else:
    with open("/home/jiashu/seq/processed/train30_test10/new_10_cases.pkl", "rb") as f:
        seq = pickle.load(f)
    df = pd.read_excel(
        data_dir / "Dissection Gestures_Clinical Data (new cases).xlsx",
        engine='openpyxl'
    )
    # only use the common 10 cases
    df = df[df['Case ID'].isin(seq.keys())]
    df['CCI'][df['CCI'].isna()] = 7


df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM",], axis=1)

###############
# model
###############

gesture_model_ckpts = [
    "/home/jiashu/seq/selected_ckpt/3667/epoch06-AUROC0.62-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3669/epoch10-AUROC0.71-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3736/epoch03-AUROC0.64-acc0.50.ckpt",
    "/home/jiashu/seq/selected_ckpt/3681/epoch09-AUROC0.74-acc0.50.ckpt",
]

attn_gesture_model_ckpts = [
    "logs/save/4502/ckpt/epoch75-AUROC0.85-acc0.60.ckpt",
    "logs/save/4503/ckpt/epoch33-AUROC0.94-acc0.60.ckpt",
    "logs/save/4504/ckpt/epoch28-AUROC0.80-acc0.50.ckpt",
    "logs/save/4505/ckpt/epoch95-AUROC0.88-acc0.50.ckpt",
]

def emb(model, packed_sequence: PackedSequence):
    embs = []
    for gesture in packed_sequence.data.unbind():
        # occlusion, use -1 for zero emb
        if gesture.item() == -1:
            embs.append(model.zero_emb)
        else:
            embs.append(model.embeddings_L(gesture))
    embs = torch.stack(embs, dim=0)
    return PackedSequence(embs, packed_sequence.batch_sizes)

@torch.no_grad()
def generate_emb(k_fold) -> float:
    if not SKIP_GESTURE:
        d = gesture_model_ckpts if not GESTURE_IS_ATTN else attn_gesture_model_ckpts
        gesture_model = BaselineModule.load_from_checkpoint(
            d[k_fold]
        ).model

        gesture_model.eval()
        gesture_model.zero_grad()
    
    if not SKIP_TABULAR:
        clinic_emb_name = "tab_transformer"
        # clinic_emb_name = "ft_transformer"
        # clinic_emb_name = "autoint"
        clinic_model = TabularModel.load_from_checkpoint(f"/home/jiashu/seq/selected_ckpt/{clinic_emb_name}/k{k_fold}")
        clinic_model = DeepFeatureExtractor(clinic_model)

    k_dir = processed_data_dir / f"k{k_fold}"

    for split in splits: 
        with open(k_dir / f"{split}_combined.pkl", "rb") as f:
            seq = pickle.load(f)
        if not SKIP_GESTURE:
            if not GESTURE_IS_ATTN:
                # i.e. LSTM model
                ds = ConcatDataset([SeqData(seq)])
                dataloader = DataLoader(
                    ds, batch_size=1,
                    collate_fn=SeqData.collate_fn, drop_last=False,
                    num_workers=1, pin_memory=True
                )
                for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
                    id = batch['id'][0]
                    
                    gesture_L = batch["L"]["gesture"]
                    emb_gesture_L: PackedSequence = emb(gesture_model, gesture_L)
                    pad_L, len_L = pad_packed_sequence(emb_gesture_L, batch_first=True)

                    gesture_R: PackedSequence = batch["R"]["gesture"]
                    emb_gesture_R: PackedSequence = emb(gesture_model, gesture_R)
                    pad_R, len_R = pad_packed_sequence(emb_gesture_R, batch_first=True)

                    # (bsz, hidden_dim)
                    h_L, _, _ = gesture_model.lstm(pad_L)

                    # (bsz, hidden_dim)
                    h_R, _, _ = gesture_model.lstm(pad_R)

                    h = torch.cat([h_L, h_R], dim=1)
                    # (1, 356)
                    h = gesture_model.linear(h)

                    seq[id]['gesture_emb'] = h.detach().cpu().numpy()

                assert all([
                    "gesture_emb" in data
                    for data in seq.values()
                ])
            else:
                # i.e. attn model
                ds = SeqAttnData(seq)
                dataloader = DataLoader(
                    ds, batch_size=len(ds),
                    collate_fn=SeqAttnData.collate_fn, drop_last=False,
                    num_workers=10, pin_memory=True
                )
                for batch in dataloader:
                    emb = gesture_model(**batch, output_emb=True)
                    for case, h in zip(batch["case"], emb):
                        # (1, 256)
                        seq[case]['attn_gesture_emb'] = h.detach().cpu().numpy().reshape(1, -1)

                assert all([
                    "attn_gesture_emb" in data
                    for data in seq.values()
                ])
        if not SKIP_TABULAR:
            idxs = df['Case ID'].isin(seq.keys()).values
            X_split = X[idxs]

            # (N, 32)
            split_emb  = clinic_model.transform(X_split).values
            split_df = pd.merge(X_split, df)
            for i, row in split_df.iterrows():
                # (1, 32)
                clinic_emb = split_emb[i].reshape(1, -1)
                id = row['Case ID']
                seq[id][clinic_emb_name] = clinic_emb
        
            assert all([
                clinic_emb_name in data
                for data in seq.values()
            ])

        with open(k_dir / f"{split}_combined.pkl", "wb") as f:
            pickle.dump(seq, f)
    
for k_fold in range(4):
    generate_emb(k_fold)
