"""
Use XGBosst to predict clinical only data (tabular)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pickle
import sys
sys.path.append("/home/jiashu/seq")
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from xgboost import XGBClassifier

from pytorch_tabular import TabularModel

k_fold = 0
data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

###############
# data
###############

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
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM",], axis=1)

###############
# model
###############
def eval_clinic(k_fold) -> float:
    # model_dir = "tab_transformer"
    # model_dir = "ft_transformer"
    # model_dir = "autoint"
    # clinic_model = TabularModel.load_from_checkpoint(f"/home/jiashu/seq/selected_ckpt/{model_dir}/k{k_fold}")
    clinic_model = XGBClassifier(
        objective="binary:logistic")
    clinic_model.load_model(f"/home/jiashu/seq/selected_ckpt/xgboost/client-k{k_fold}.json")

    # if tabnet
    # pred_df = clinic_model.predict(X)
    # prob = pred_df['1_probability']
    # return roc_auc_score(X['ESI @ 12 mo'], prob)

    # if xgbost
    y = df['ESI @ 12 mo']
    XX = X.drop(['ESI @ 12 mo'], axis=1)
    prob = clinic_model.predict_proba(XX)
    return roc_auc_score(y, prob[:, 1])

    
rets = []
for k_fold in range(4):
    rets.append(eval_clinic(k_fold))
rets = np.array(rets)
rets = np.where(rets > 0.5, rets, 1-rets)
print(rets, rets.mean(), rets.std(ddof=1))
