
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pytorch_tabular.tabular_model import TabularModel
import torch
from torch.utils.data.dataloader import DataLoader
from src.data.SeqAttnDataModule import SeqAttnData

from src.module.BaselineModule import BaselineModule
from src.module.model.AttnModel import AttnModel


attn_gesture_model_ckpts = [
    "logs/save/4502/ckpt/epoch75-AUROC0.85-acc0.60.ckpt",
    "logs/save/4503/ckpt/epoch33-AUROC0.94-acc0.60.ckpt",
    "logs/save/4504/ckpt/epoch28-AUROC0.80-acc0.50.ckpt",
    "logs/save/4505/ckpt/epoch95-AUROC0.88-acc0.50.ckpt",
]

data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

with open("/home/jiashu/seq/processed/train30_test10/new_10_cases.pkl", "rb") as f:
    seq = pickle.load(f)
df = pd.read_excel(
    data_dir / "Dissection Gestures_Clinical Data (new cases).xlsx",
    engine='openpyxl'
)
# only use the common 10 cases
df = df[df['Case ID'].isin(seq.keys())]

df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
df['CCI'][df['CCI'].isna()] = 7
X = df.drop(["Surgeon ID", "Pre-Op SHIM",], axis=1)

for k, gckpt in enumerate(attn_gesture_model_ckpts):
    gesture_mapping = {}
    with open(f"/home/jiashu/seq/processed/train30_test10/k{k}/gesture_mapping.txt") as f:
        for line in f:
            gesture, gid = line.rstrip().split("->")
            gesture_mapping[gesture] = int(gid)
    gestures_clone = deepcopy(seq)
    for LR in ["L", "R"]:
        for case in gestures_clone:
            gestures_clone[case][LR]["gesture"] = np.array([
                gesture_mapping[value]
                        for value in gestures_clone[case][LR]['gesture']
            ])
    dataset = SeqAttnData(gestures_clone)
    dataloader = DataLoader(
        dataset, batch_size=len(dataset),
        collate_fn=SeqAttnData.collate_fn, drop_last=False,
        num_workers=10, pin_memory=True
    )
    
    model: AttnModel = BaselineModule.load_from_checkpoint(gckpt).model
    model.eval()
    model.zero_grad()

    with torch.no_grad():
        for batch in dataloader:
            logits = model(**batch)
            labels = batch["label"]
            cases = batch["case"]

    model_dir = "ft_transformer"
    clinic_model = TabularModel.load_from_checkpoint(f"/home/jiashu/seq/selected_ckpt/{model_dir}/k{k}")
    case2 = X['Case ID']
    pred_df = clinic_model.predict(X.drop(['Case ID'], axis=1))
    prob = pred_df['1_probability']
    label = X['ESI @ 12 mo']

    case2info = {}
    for case, glogit, label in zip(cases, logits, labels):
        cprob = prob[ case2 == case ].values[0]
        case2info[case] = (
            glogit.item(), cprob, label.item()
        )
    with open(f"{k}-conf-matrix.pkl", "wb") as f:
        pickle.dump(case2info, f)


