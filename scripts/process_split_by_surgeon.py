import os
import pandas as pd
from pathlib import Path
import pickle, json
import random

THRESHOLD = 3

if __name__ == "__main__":
    random.seed(42)

    # 1. load all cases data
    data_dir = Path("/home/jiashu/seq/raw_data")
    output_file = Path("/home/jiashu/seq/processed")
    seq = {}
    for split in ["train", "valid", "test"]:
        with open(output_file / "train30_test10" / "k0" / f"{split}_processed.pkl", "rb") as f:
            data = pickle.load(f)
        seq = {**data, **seq}


    # 2. split by surgeon
    df = pd.read_excel(data_dir / "clinical data_50 cases included.xlsx", engine="openpyxl")
    surgeons = set(pd.value_counts(df['Left NVB Surgeon']).index)
    output_file = output_file / "split_by_surgeon"
    os.makedirs(output_file, exist_ok=True)

    groups = df.groupby("Left NVB Surgeon")
    stats = {}
    for i, (surgeon_id, group) in enumerate(groups):
        print(f"surgeon {surgeon_id}")
        cases = group["Case ID"].values
        other_surgeons = surgeons - {surgeon_id}

        # select valid set
        while True:
            sample = random.choice(list(other_surgeons))
            sdf = df.query("`Left NVB Surgeon` == @sample")
            if sdf.shape[0] >= THRESHOLD:
                break

        valid_cases = sdf["Case ID"].values
        test_set = {
            k: v for k, v in seq.items()
            if k in cases
        }
        valid_set = {
            k: v for k, v in seq.items()
            if k in valid_cases
        }
        train_set = {
            k: v for k, v in seq.items()
            if (k not in cases) and (k not in valid_cases)
        }
        assert len(train_set) + len(valid_set) + len(test_set) == len(seq) == 50

        # (surgeon (list), case ID)
        stats[i] = {
            "train": (list(other_surgeons - {sample}), list(train_set.keys())),
            "valid":([sample], list(valid_set.keys())),
            "test": ([surgeon_id], list(test_set.keys())),
        }

        os.makedirs(output_file / f"k{i}", exist_ok=True)

        for data, split in zip([train_set, valid_set, test_set], ["train", "valid", "test"]):
            with open(output_file / f"k{i}" / f"{split}_processed.pkl", "wb") as f:
                pickle.dump(data, f)
        
    with open(output_file / "split_stat.jsonl", "w") as f:
        json.dump(stats, f)