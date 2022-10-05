"""
Use XGBosst to predict clinical only data (tabular)
"""

from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

df = pd.read_excel(
    data_dir / "Dissection Gesture_Clinical Data.xlsx"
)

df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
y = df['ESI @ 12 mo']
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM", 'ESI @ 12 mo'], axis=1)

def auc_for_k_fold(clf, X, y, k_fold: int):
    """
    return acc for this k_fold
    if plot=True:
        plot feature importance i
        save model
    """
    k_dir = processed_data_dir / f"k{k_fold}"
    with open(k_dir / "train.pkl", "rb") as f:
        seq = pickle.load(f)
    train_idxs = df['Case ID'].isin(seq.keys()).values
    X_train, y_train = X[train_idxs], y[train_idxs]
    
    with open(k_dir / "valid.pkl", "rb") as f:
        seq = pickle.load(f)
    test_idxs = df['Case ID'].isin(seq.keys()).values
    X_test, y_test = X[test_idxs], y[test_idxs]
    
    clf.fit(
        X_train, y_train,
    )
    # auc
    prob = clf.predict_proba(X_test)
    ret = roc_auc_score(y_test, prob[:, 1])

    return ret

# save feature importance
for k_fold in range(4):
    for model in [
        SVC(gamma='auto', probability=True),
        LogisticRegression()
    ]:
        clf = make_pipeline(StandardScaler(), model)
        auc = auc_for_k_fold(clf, X, y, k_fold=k_fold)
        print(f"k_fold={k_fold}, model={model} => {auc}")
