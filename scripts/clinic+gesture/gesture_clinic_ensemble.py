"""
Use XGBosst to predict clinical only data (tabular)
"""

from torch.nn.utils.rnn import PackedSequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
import tqdm
from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score
import optuna, os
import numpy as np
import matplotlib.pyplot as plt
from src.data.SeqData import SeqData

from src.module.BaselineModule import BaselineModule

data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

df = pd.read_excel(
    data_dir / "Dissection Gesture_Clinical Data.xlsx",
    engine='openpyxl'
)

df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
y = df['ESI @ 12 mo']
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM", 'ESI @ 12 mo'], axis=1)


gesture_model_ckpts = [
    "/home/jiashu/seq/selected_ckpt/3667/epoch06-AUROC0.62-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3669/epoch10-AUROC0.71-acc0.60.ckpt",
    "/home/jiashu/seq/selected_ckpt/3736/epoch03-AUROC0.64-acc0.50.ckpt",
    "/home/jiashu/seq/selected_ckpt/3681/epoch09-AUROC0.74-acc0.50.ckpt",
]

# save feature importance
for k_fold in range(4):
    clinic_model = XGBClassifier(
        objective="binary:logistic")
    clinic_model.load_model(f"/home/jiashu/seq/selected_ckpt/xgboost/client-k{k_fold}.json")

    gesture_model = BaselineModule.load_from_checkpoint(
        gesture_model_ckpts[k_fold]
    ).model

    gesture_model.eval()
    gesture_model.zero_grad()

    k_dir = processed_data_dir / f"k{k_fold}"


    with open(k_dir / f"valid_processed.pkl", "rb") as f:
        seq = pickle.load(f)
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

    idxs = df['Case ID'].isin(seq.keys()).values
    X_split = X[idxs]

    # (N, 32)
    split_emb  = clinic_model.transform(X_split).values
    split_df = pd.merge(X_split, df)
    for i, row in split_df.iterrows():
        # (1, 32)
        clinic_emb = split_emb[i].reshape(1, -1)
        id = row['Case ID']
        seq[id]['clinic_emb'] = clinic_emb
    
    assert all([
        "gesture_emb" in data and "clinic_emb" in data
        for data in seq.values()
    ])
    with open(k_dir / f"{split}_combined.pkl", "wb") as f:
        pickle.dump(seq, f)