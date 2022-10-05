
from pathlib import Path
import pickle
from tqdm import tqdm

import torch

from gesture_clinic_mlp import load_emb, MLP


processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")
GMODEL = "gesture_emb"
# CMODEL = "ft_transformer"
CMODEL = "tab_transformer"
device = torch.device("cpu")

for k in tqdm(range(4)):
    print(k)
    if GMODEL == "gesture_emb":
        model: MLP = torch.load(
                    f"/home/jiashu/seq/selected_ckpt/combined/{k}/{CMODEL}.pt").to(device).eval()
    
        for split in ["train", "valid", "test"]:
            X, y, cases = load_emb(k, split, GMODEL, CMODEL, return_case=True)

            with torch.no_grad():
                emb = None
                def hook(module, input, output):
                    global emb
                    emb = output
                model.mlp[5].register_forward_hook(hook)

                model(X)
                assert emb is not None

            with open(processed_data_dir / f"k{k}" / f"{split}_combined.pkl", "rb") as f:
                data = pickle.load(f)
            
            for case, e in zip(cases, emb):
                data[case][f"IMV-{CMODEL}"] = e.detach().cpu().numpy()

            with open(processed_data_dir / f"k{k}" / f"{split}_combined.pkl", "wb") as f:
                pickle.dump(data, f)
