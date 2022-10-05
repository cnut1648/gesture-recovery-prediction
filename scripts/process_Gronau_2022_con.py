"""
additional Gronau cases
add to existing train30_test10/k0
use con_3M as label
"""

from collections import defaultdict
import glob
import os
from pathlib import Path
import pickle
from copy import deepcopy
from shutil import copy2
import shutil
from typing import DefaultDict
import numpy as np

import pandas as pd
import math


def format_arm(value) -> str:
    if value is np.nan:
        return "nan"
    elif type(value) is str:
        return value.strip()
    raise NotImplementedError

def format_note(value) -> str:
    if value is np.nan:
        return "nan"
    elif type(value) is float and np.isnan(value):
        return "nan"
    elif type(value) is str:
        return value.strip()
    raise NotImplementedError

data_dir = Path("/home/jiashu/seq/raw_data")
out_dir = Path("/home/jiashu/seq/processed") / "USC+Gronau_con" / "k0"

# data: dict of. seq of gesture, note, arm
# P-059: {L: data, R: data}
seq = defaultdict(lambda: {})

for path in glob.glob(str(data_dir / "Gronau" / "*.xlsx")):
    excel = pd.ExcelFile(path, engine="openpyxl")
    for sheet in excel.sheet_names:
        sheet_name = sheet.strip()
        assert sheet_name[:2] == "GP"

        # GP-007_NS_L to GP-007_L
        if "NS_" in sheet_name:
            sheet_name = sheet_name.replace("NS_", "")
        caseid, LR = sheet_name.split("_")
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        # weird case
        if (
            (caseid == "GP-007" and LR == "R")
            or 
            (caseid == "GP-008" and LR == "L")
            or 
            (caseid == "GP-041" and LR == "L")
            or 
            (caseid == "GP-041" and LR == "R")
            or 
            (caseid == "GP-014" and LR == "L")
        ):
            df = df.dropna(how="all")
        data = {
            "gesture": df["Gesture"].str.strip().tolist(),
            "arm": df["Arm"].apply(format_arm).tolist(),
            "note": df["Note"].apply(format_note).tolist()
        }
        seq[caseid][LR] = data

seq = {
    caseid: LR
    for caseid, LR in seq.items()
    if "L" in LR and "R" in LR
}
assert len(seq) == 30

labels = pd.read_excel("/home/jiashu/seq/raw_data/clinical data_80 cases included.xlsx", engine="openpyxl")
for case in seq:
    labels_row = labels.query('`Case ID` == @case')
    # has 1 and only 1 row
    assert labels_row.shape[0] == 1
    label = labels_row['Con_3M'].values[0]
    seq[case]['label'] = label



os.makedirs(out_dir, exist_ok=True)
orig_dir = Path("/home/jiashu/seq/processed") / "train30_test10" / "k0"
copy2(orig_dir/ "gesture_mapping.txt", out_dir/ "gesture_mapping.txt")
for split in ["train", "valid", "test"]:
    copy2(orig_dir / f"{split}_combined.pkl", out_dir / f"{split}_combined.pkl")


gesture_mapping = {}
with open(out_dir / "gesture_mapping.txt") as f:
    for line in f:
        gesture, gid = line.rstrip().split("->")
        gesture_mapping[gesture] = int(gid)

for case in seq:
    for LR in ["L", "R"]:
        seq[case][LR]['gesture'] = np.array([
            gesture_mapping[value] if value != "a, m" else gesture_mapping["m"]
            for value in seq[case][LR]['gesture']
            if value != "" and type(value) is str
        ])
with open(out_dir / "test_combined.pkl", "rb") as f:
    newseq = pickle.load(f)
newseq = {
    **seq,
    **newseq
}
with open(out_dir / "test_combined.pkl", "wb") as f:
    pickle.dump(newseq, f)


for split in ["train", "valid", "test"]:
    with open(out_dir / f"{split}_combined.pkl", "rb") as f:
        newseq = pickle.load(f)
    to_delete = []
    for case in newseq:
        labels_row = labels.query('`Case ID` == @case')
        label = labels_row['Con_3M'].values[0]
        newseq[case]['label'] = label
        if case in ["P-103", "P-203", "GP-032"]:
            to_delete.append(case)
    for rm in to_delete:
        newseq.pop(rm)
    with open(out_dir / f"{split}_combined.pkl", "wb") as f:
        pickle.dump(newseq, f)

# USC only
out_dir2 = Path("/home/jiashu/seq/processed") / "USC_con" / "k0"

shutil.copytree(out_dir, out_dir2, dirs_exist_ok=True)

with open(out_dir2 / f"test_combined.pkl", "rb") as f:
    newseq = pickle.load(f)
for case in list(newseq.keys())[:]:
    if case.startswith("GP"):
        newseq.pop(case)
with open(out_dir2 / f"test_combined.pkl", "wb") as f:
    pickle.dump(newseq, f)