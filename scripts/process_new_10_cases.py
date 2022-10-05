"""
additional 10 new cases for out-sample testing
"""

from pathlib import Path
import pickle
from copy import deepcopy
from typing import DefaultDict
import numpy as np

import pandas as pd


data_dir = Path("/home/jiashu/seq/raw_data")
labels = pd.read_excel(
    data_dir / "Dissection Gestures_Clinical Data (new cases).xlsx",
    engine='openpyxl'
)

gestures = DefaultDict(dict)
cases = pd.ExcelFile(
    data_dir / "10 more ESI cases annotation.xlsx",
    engine='openpyxl'
)

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


for sheet in cases.sheet_names:
    case, LR = sheet.split(" ")
    df = pd.read_excel(
        data_dir / "10 more ESI cases annotation.xlsx",
        sheet_name=sheet,
        engine='openpyxl'
    )
    data = {
        "gesture": df["Gesture"].str.strip().tolist(),
        "arm": df["Arm"].apply(format_arm).tolist(),
        "note": df["Note"].apply(format_note).tolist()
    }

    gestures[case][LR] = data

for case in gestures:
    labels_row = labels.query('`Case ID` == @case')
    # has 1 and only 1 row
    assert labels_row.shape[0] == 1
    label = labels_row['ESI @ 12 mo'].values[0]
    gestures[case]['label'] = label

with open("/home/jiashu/seq/processed/train30_test10/new_10_cases.pkl", "wb") as f:
    pickle.dump(gestures, f)

for kfold in range(4):
    gesture_mapping = {}
    with open(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/gesture_mapping.txt") as f:
        for line in f:
            gesture, gid = line.rstrip().split("->")
            gesture_mapping[gesture] = int(gid)
    gestures_clone = deepcopy(gestures)
    for case in gestures_clone:
        for LR in ["L", "R"]:
            gestures_clone[case][LR]['gesture'] = np.array([
                gesture_mapping[value]
                for value in gestures_clone[case][LR]['gesture']
            ])
    with open(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/test_processed.pkl", "wb") as f:
        pickle.dump(gestures_clone, f)