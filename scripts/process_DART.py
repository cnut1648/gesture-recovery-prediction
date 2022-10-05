from collections import defaultdict
import pandas as pd
import pickle

df = pd.read_excel(
    "/home/jiashu/seq/raw_data/SPSS.xlsx",
    engine='openpyxl'
)

# filter to 40 cases
with open("/home/jiashu/seq/processed/train30_test10/k0/train.pkl", "rb") as f1, open("/home/jiashu/seq/processed/train30_test10/k0/valid.pkl", "rb") as f2:
    seq = {
        **pickle.load(f1),
        **pickle.load(f2),
    }

cases = defaultdict(dict)
vids = df['Video']
for row_no, vid in enumerate(vids):
    if "_" in vid:
        id, LR = vid.split("_")
    else:
        id, LR = vid.split(" ")
    if id in seq:
        cases[id][LR] = row_no

assert len(cases) == 40 and all([
    LR in values
    for values in cases.values()
    for LR in ["L", "R"]
])

## create data
DART_SCORES = ["gs", "iva", "rtp", "th", "tr", "e"]
all_dart_cols = [
    f"{dart}{num}"
    for dart in DART_SCORES
    for num in range(1, 6)
]
assert all(col in df.columns for col in all_dart_cols)


def combine_LR_case_DART_score(kfold, split):
    """
    merge LR DART scores of the same video case to the same row
    """
    with open(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/{split}.pkl", "rb") as f:
        seq = pickle.load(f)
        load_cases = {
            case: seq[case]['label']
            for case in seq.keys()
        }
    
    rows = []
    for case, label in load_cases.items():
        left = df.loc[cases[case]['L']][all_dart_cols].tolist()
        right = df.loc[cases[case]['R']][all_dart_cols].tolist()
        rows.append(
            [case, label] + left + right
        )
    
    X = pd.DataFrame(rows, 
        columns=[
            "case", "label"] + list(map(lambda c: c+" L", all_dart_cols)) + list(map(lambda c: c+" R", all_dart_cols))
    )
    return X

aucs = []
for kfold in range(4):
    for split in ['train', 'valid']:
        X = combine_LR_case_DART_score(kfold, split)
        X.to_csv(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/{split}-DART.csv")