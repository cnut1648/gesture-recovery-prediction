"""
eval gesture+clinic model in new 10 cases
"""

from pathlib import Path
import pickle
import numpy as np

import torch
import sys; sys.path.insert(0, ".")
from gesture_clinic_mlp import MLP, load_emb 
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    # device = torch.device("cuda")
    device = torch.device("cpu")
    # CMODEL = "autoint"
    # CMODEL = "ft_transformer"
    GMODEL = "attn_gesture_emb"
    CMODEL = "tab_transformer"
    rets = []
    with torch.no_grad():
        for kfold in range(4):
            model = torch.load(
                f"/home/jiashu/seq/selected_ckpt/combined/{kfold}/{GMODEL}-{CMODEL}.pt").to(device).eval()
            X, y = load_emb(kfold, 'test', GMODEL, CMODEL)

            logits = model(X).view(-1)
            auc = roc_auc_score(y.cpu().numpy(), logits.detach().cpu().numpy())
            rets.append(auc)
    rets = np.array(rets)
    rets = np.where(rets > 0.5, rets, 1-rets)
    print(rets, rets.mean(), rets.std(ddof=1))
