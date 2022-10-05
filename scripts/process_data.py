import glob, os
from typing import Optional
import pandas as pd
from pathlib import Path
from collections import defaultdict
from math import isnan
import pickle
import numpy as np

data_dir = "/home/jiashu/seq/raw_data"
data_dir = Path(data_dir)
output_file = "/home/jiashu/seq/processed/"
# k_fold: Optional[int] = None
k_fold: Optional[int] = 4
# train 30 + test 10
# otherwise train 24 + val 8 + test 8
split_40_10 = True
if split_40_10 == True:
    assert k_fold == 4
else:
    assert k_fold == 5

# data: dict of. seq of gesture, note, arm
# P-059: {L: data, R: data}
seq = defaultdict(lambda: {})

for name, LR in seq.items():
    try:
        assert len(LR) == 2 and "L" in LR and "R" in LR
    except:
        print(name)

def format_arm(value) -> str:
    if value is np.nan:
        return "nan"
    elif type(value) is str:
        return value.strip()
    raise NotImplementedError

def format_note(value) -> str:
    if value is np.nan:
        return "nan"
    elif type(value) is float and isnan(value):
        return "nan"
    elif type(value) is str:
        return value.strip()
    raise NotImplementedError


for path in glob.glob(str(data_dir / "annotations" / "*.xlsx")):
    excel = pd.ExcelFile(path)
    for sheet in excel.book.sheets():
        sheet_name = sheet.name.strip()
        # eg change 122-L to P122-L
        if sheet_name[0].isdigit():
            name, LR = sheet_name.split("-")
            sheet_name = f"P-{name} {LR}"

        # eg sheet P-059 L
        caseid, LR = sheet_name.split()
        df = pd.read_excel(path, sheet_name=sheet.name)
        data = {
            "Gesture": df["Gesture"].str.strip().tolist(),
            "Arm": df["Arm"].apply(format_arm).tolist(),
            "Note": df["Note"].apply(format_note).tolist()
        }
        seq[caseid][LR] = data

# rm caseid where there is no L or R
seq = {
    caseid: LR
    for caseid, LR in seq.items()
    if "L" in LR and "R" in LR
}
# append score
scores = pd.read_excel(data_dir / "dart scores.xlsx")
for _, row in scores.iterrows():
    video = row["Video"]
    caseid, LR = video.split("_")
    if caseid in seq:
        seq[caseid]["scores"] = {
            "ave_gs": row["ave_gs"],
            "ave_iva": row["ave_iva"],
            "ave_rtp": row["ave_rtp"],
            "ave_th": row["ave_th"],
            "ave_tr": row["ave_tr"],
        }

# append label + features
# caseid -> {"label": 0 or 1, "L": data "R": data}
# data: dict of. seq of gesture, note, arm, nerve sparing
meta = pd.read_excel(data_dir / "clinical data.xlsx")
for _, row in meta.iterrows():
    caseid = row["Case ID"]
    # ignore caseid that has either no L or R
    if caseid in seq:
        seq[caseid]["label"] = row["ESI @ 12 mo"]
        seq[caseid]["L"]["Nerve Sparing"] = row["Nerve Sparing- Left (0=no, 1=partial, 2= full)"]
        seq[caseid]["R"]["Nerve Sparing"] = row["Nerve Sparing- Right (0=no, 1=partial, 2= full)"]

assert len(seq) == 40, "should have 40 cases in total"

if not k_fold:
    with open(os.path.join(output_file, "seq.pkl"), "wb") as f:
        pickle.dump(seq, f)
else:
    # k fold 
    assert type(k_fold) is int
    from sklearn.model_selection import StratifiedKFold


    skf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=False)

    caseids = []; labels = []
    for caseid, data in seq.items():
        caseids.append(caseid)
        labels.append(data["label"])
    caseids = np.array(caseids)
    labels = np.array(labels)

    size = len(caseids) // k_fold

    # train 30 + test 10
    if split_40_10:
        for k, (train_idx, test_idx) in enumerate(skf.split(caseids, labels)):
            k_fold_dir = os.path.join(output_file, "train30_test10", f"k{k}")
            os.makedirs(k_fold_dir, exist_ok=True)
            print(f"{k} -> {len(train_idx)} + {len(test_idx)}")

            for phase, idxs in zip(["train", "valid"], [train_idx, test_idx]):
                with open(os.path.join(k_fold_dir, f"{phase}.pkl"), "wb") as f:
                    split_seq = {
                        k: v for k, v in seq.items()
                        if k in caseids[idxs]
                    }
                    pickle.dump(split_seq, f)

            assert len(os.listdir(k_fold_dir)) == 2

    # train 24 + val 8 + test 8
    else:
        for k, (train_idx, val_idx) in enumerate(skf.split(caseids, labels)):
            test_idx, train_idx = train_idx[:size], train_idx[size:]
            k_fold_dir = os.path.join(output_file, f"k{k}")
            os.makedirs(k_fold_dir, exist_ok=True)
            print(f"{k} -> {len(train_idx)} + {len(val_idx)} + {len(test_idx)}")

            for phase, idxs in zip(["train", "valid", "test"], [train_idx, val_idx, test_idx]):
                with open(os.path.join(k_fold_dir, f"{phase}.pkl"), "wb") as f:
                    split_seq = {
                        k: v for k, v in seq.items()
                        if k in caseids[idxs]
                    }
                    pickle.dump(split_seq, f)

            assert len(os.listdir(k_fold_dir)) == 3