from pathlib import Path
import os, sys
import glob


import pickle
import numpy as np
import torch
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict
from copy import deepcopy
import sys; sys.path.insert(0, "/home/jiashu/seq")
print(sys.path)

from src.module.BaselineModule import BaselineModule
from src.data.SeqAttnDataModule import SeqAttnData
from src.module.model.AttnModel import AttnModel

def get_all_gesture_attn_dataloader(data_dir, k):
    ds = {}
    for split in ["train", "valid"]:
        file = data_dir / f"k{k}" / f"{split}_processed.pkl"
        with open(file, "rb") as f:
            seq = pickle.load(f)
        ds[split] = (SeqAttnData(seq), seq)

    with open(data_dir / "new_10_cases.pkl", "rb") as f:
        seq = pickle.load(f)
    
    gesture_mapping = {}
    with open(data_dir / f"k{k}" / "gesture_mapping.txt") as f:
        for line in f:
            gesture, gid = line.rstrip().split("->")
            gesture_mapping[gesture] = int(gid)

    for LR in ["L", "R"]:
        for case in seq:
            seq[case][LR]["gesture"] = np.array([
                gesture_mapping[value]
                        for value in seq[case][LR]['gesture']
            ])
    ds["test"] = (SeqAttnData(seq), seq)


    for split, (dataset, seq) in ds.items():
        ds[split] = (
            DataLoader(
                dataset, batch_size=len(dataset),
                collate_fn=SeqAttnData.collate_fn, drop_last=False,
                num_workers=10, pin_memory=True
            ), seq)
    return ds


if __name__ == "__main__":
    # [14246, 14275, 14260, 14307, 14300]
    # ckpts = [
    #     "logs/save/14246/ckpt/epoch05-AUROC0.92-acc0.60.ckpt",
    #     "logs/save/14275/ckpt/epoch19-AUROC1.00-acc0.50.ckpt",
    #     "logs/save/14260/ckpt/epoch34-AUROC0.96-acc0.60.ckpt",
    #     "logs/save/14307/ckpt/epoch12-AUROC0.92-acc0.60.ckpt",
    #     "logs/save/14300/ckpt/epoch02-AUROC1.00-acc0.70.ckpt",
    # ]
    # ckpts = [
    #     # "logs/save/14306/ckpt/epoch44-AUROC1.00-acc0.50.ckpt",
    #     # "logs/save/14332/ckpt/epoch23-AUROC0.84-acc0.80.ckpt",
    #     # "logs/save/14296/ckpt/epoch06-AUROC0.96-acc0.50.ckpt",
    #     # "logs/save/14253/ckpt/epoch26-AUROC0.96-acc0.40.ckpt",
    #     "logs/save/14336/ckpt/epoch04-AUROC0.83-acc0.60.ckpt",
    # ]
    out = open("extract.txt", "w")
    ckpts = [
        ckpt
        for id in range(14339, 14438+1)
        for ckpt in glob.glob(f"logs/save/{id}/ckpt/**.ckpt", recursive=True)
    ]
    print("len of ckpts = ", len(ckpts))
    artifact_dir = Path("artifact/transformer/")
    os.makedirs(artifact_dir, exist_ok=True)
    data_dir = Path("processed/train30_test10/")


    k = 0
    ds = get_all_gesture_attn_dataloader(data_dir, k)
    seq= {}
    for split, (dataloader, s) in ds.items():
        for case in s:
            s[case]["split"] = split
        seq = {**seq, **s}
    ds = ConcatDataset([d.dataset for _, (d, _) in ds.items()])
    dataloader = DataLoader(
        ds, batch_size=len(ds),
        collate_fn=SeqAttnData.collate_fn, drop_last=False,
        num_workers=10, pin_memory=True
    )
    iterd = iter(dataloader)
    batch = next(iterd)
    # assert exhaust
    try:
        batch2 = next(iterd)
        sys.exit(1)
    except:
        pass

    print("begin")
    with torch.no_grad():
        for k, ckpt in enumerate(ckpts):
            print(ckpt)
            out.write(str(ckpt) + "\n")
            out.flush()
            model: AttnModel = BaselineModule.load_from_checkpoint(ckpt).model
            model.eval()
            model.zero_grad()
            # {
            #     case => 
            #      Key[attn_scores, label, split, orig_logits, orig_gesture]
            #      Value[Dict[first/avg -> (L, R)], 1/0, str, float, (L, R)]
            # }}
            attns = defaultdict(dict)

            # (bsz, L, L)
            (first_L, avg_L), (first_R, avg_R) = model.get_scores(**batch)
            logits = model(**batch)
            labels = batch["label"]
                        
            for case, fL, aL, fR, aR, logit, label in zip(
                batch["case"], 
                first_L, avg_L, 
                first_R, avg_R, 
                logits, labels
            ):
                orig_L = seq[case]["L"]["gesture"]
                orig_R = seq[case]["R"]["gesture"]
                attns[case]['orig_gesture'] = (orig_L, orig_R)
                # renormalize based on [CLS]'s attn score
                attns[case]["attn_scores"] = {
                    "first": (fL[0][1:len(orig_L)+1].softmax(0).detach().cpu().numpy(), fR[0][1:len(orig_L)+1].softmax(0).detach().cpu().numpy()),
                    "avg": (aL[0][1:len(orig_L)+1].softmax(0).detach().cpu().numpy(), aR[0][1:len(orig_L)+1].softmax(0).detach().cpu().numpy()),
                }
                # layer_attns[case]["pred_label"] = (logit > best_thr).item()
                attns[case]["split"] = seq[case]["split"]
                attns[case]["label"] = label.item()
                attns[case]["orig_logits"] = logit.item()

            labels = [
                d["label"]
                for d in attns.values()
            ]
            logits = [
                d["orig_logits"]
                for d in attns.values()
            ]
            try:
                _, _, thresholds = roc_curve(labels, logits)
                best_thr, best_acc = None, -1
                for thr in thresholds:
                    acc = accuracy_score(logits >= thr, labels)
                    if acc > best_acc: 
                        best_acc = acc
                        best_thr = thr
                print(f"best acc is ", best_acc)
                out.write(f"best acc is " + str( best_acc) + "\n")
                out.flush()
                for case, d in attns.items():
                    logit = d["orig_logits"]
                    del d["orig_logits"]
                    d["pred_label"] = (logit >= best_thr).item()

                with open(artifact_dir / f"{ckpt.split('/')[2]}-{best_acc}.pkl", "wb") as f:
                    pickle.dump(dict(attns), f)
            except Exception as e:
                print("raise error", e)
    out.close()