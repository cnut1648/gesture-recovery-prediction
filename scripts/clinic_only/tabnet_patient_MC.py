"""
Use XGBosst to predict clinical only data (tabular)
"""

import pandas as pd
from pathlib import Path
import pickle, csv, json
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, accuracy_score, f1_score, recall_score, precision_score
import optuna, os
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from pytorch_tabular.models.autoint import AutoIntConfig

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
import sys

MODEL = sys.argv[1]
assert MODEL in ["FTtransformer", "tabtransformer", "autoint"]

data_dir = Path("/home/jiashu/seq/raw_data")

# df = pd.read_excel(
#     data_dir / "clinical data_50 cases included.xlsx",
#     engine='openpyxl'
# )
df = pd.read_excel(
    data_dir / "clinical data_80 cases included.xlsx",
    engine='openpyxl'
)

df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
df['CCI'][df['CCI'].isna()] = 7
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM", "Cohort", "Left NVB Surgeon", "Right NVB Surgeon"], axis=1)

data_config = DataConfig(
    target=['ESI @ 12 mo'],
    continuous_cols=["Age", "BMI", "CCI", "PSA", "Prostate volume (g)", "Pre-op Gleason", "Post-op Gleason"],
    categorical_cols=[ "ASA", "Nerve Sparing", "ECE", 'Radiation after surgery 1=Yes, 0=No', 'Postop ADT'],
)

trainer_config = TrainerConfig(
    auto_lr_find=False, # Runs the LRFinder to automatically derive a learning rate
    batch_size=128,
    deterministic=True,
    # disable ckpt
    checkpoints=None,
    max_epochs=15,
    gpus=1, #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

f = open(f"{MODEL}.txt", "w", newline='')
writer = csv.writer(f, delimiter="\t")

def train_model(model_config, k_fold) -> float:
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    random.seed(k_fold)
    indexes = list(range(len(X)))
    random.shuffle(indexes)
    

    # if 50 cases
    # X_train = X.iloc[indexes[:40], :]

    # X_test = X.iloc[indexes[40:], :]

    # if 80 cases
    X_train = X.iloc[indexes[:64], :]

    X_test = X.iloc[indexes[64:], :]

    tabular_model.fit(train=X_train, validation=X_test)
    pred_df = tabular_model.predict(X_test)
    prob = pred_df['1_probability']
    preds = pred_df["prediction"]
    labels = X_test['ESI @ 12 mo']

    fpr, tpr, thresholds = precision_recall_curve(X_test['ESI @ 12 mo'], prob)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    acc = accuracy_score(labels, preds)
    AUPRC = auc(tpr, fpr)
    AUROC = float(roc_auc_score(X_test['ESI @ 12 mo'], prob))
    writer.writerow(list(map(str, [AUROC, AUPRC, acc, f1, precision, recall])))
    f.flush()
    return AUROC

def get_auc_over_k_folds(model_config):
    rets = []
    for k_fold in range(50):
        auc = train_model(model_config, k_fold)
        rets.append(auc)
    return np.mean(rets)

def objective_tabtransformer(trial):
    emb_dim = trial.suggest_categorical("input_embed_dim", [16, 32, 64, 128, 256])
    emb_dropout = trial.suggest_float("embedding_dropout", 0.1, 0.6)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    share_emb = trial.suggest_categorical("share_embedding", [True, False])
    ff = trial.suggest_categorical("out_ff_layers", ["128-64-32", "64-32-16", "128-128-32"])
    bn_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    args = dict(
        input_embed_dim=emb_dim,
        embedding_dropout=emb_dropout,
        learning_rate=learning_rate,
        share_embedding=share_emb,
        out_ff_layers=ff,
    )
    if bn_norm:
        args = {
            **args,
            "use_batch_norm": True,
            "batch_norm_continuous_input": True
        }
    
    model_config = TabTransformerConfig(
        task="classification",
        metrics=["accuracy"],
        seed=42,
        **args
    )

    return get_auc_over_k_folds(model_config)

def objective_fttransformer(trial):
    args = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        input_embed_dim=trial.suggest_categorical("input_embed_dim", [16, 32, 64, 128, 256]),
        embedding_dropout=trial.suggest_float("embedding_dropout", 0.1, 0.6),
        share_embedding=trial.suggest_categorical("share_embedding", [True, False]),
        num_attn_blocks=trial.suggest_int("num_attn_blocks", 2, 10),
        attn_dropout=trial.suggest_float("attn_dropout", 0.1, 0.6),
    )
    model_config = FTTransformerConfig(
        task="classification",
        metrics=["accuracy"],
        seed=42,
        **args
    )

    return get_auc_over_k_folds(model_config)

def objective_autoint(trial):
    args = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        attn_embed_dim=trial.suggest_categorical("attn_embed_dim", [8, 16, 32]),
        num_attn_blocks=trial.suggest_int("num_attn_blocks", 2, 5),
        attn_dropouts=trial.suggest_float("attn_dropouts", 0.1, 0.6),
        has_residuals=trial.suggest_categorical("has_residuals", [True, False]),
        embedding_dim=trial.suggest_categorical("embedding_dim", [16, 32, 64]),
        embedding_dropout=trial.suggest_float("embedding_dropout", 0.1, 0.6),
        deep_layers=trial.suggest_categorical("has_residuals", [True, False]),
    )
    model_config = AutoIntConfig(
        task="classification",
        metrics=["accuracy"],
        seed=42,
        **args
    )

    return get_auc_over_k_folds(model_config)



models_objectives = {
    "tabtransformer": objective_tabtransformer,
    "FTtransformer": objective_fttransformer,
    "autoint": objective_autoint
}

print(f"model = {MODEL}, using objective {models_objectives[MODEL]}")

study = optuna.create_study(direction='maximize')
study.optimize(models_objectives[MODEL], n_trials=300)
# study.optimize(models_objectives[MODEL], n_trials=1)


trial = study.best_trial

f.write(f"best study has avg acc over k-fold = {trial.value}\n")
f.close()
args = trial.params
with open(f"{MODEL}_BEST.json", "w") as f:
    json.dump(args, f)

if MODEL == "tabtransformer":
    config = TabTransformerConfig

    bn_norm = args.pop("use_batch_norm")
    if bn_norm:
        args = {
            **args,
            "use_batch_norm": True,
            "batch_norm_continuous_input": True
        }
elif MODEL == "FTtransformer":
    config = FTTransformerConfig
else: # MODEL == "autoint":
    config = AutoIntConfig

f = open(f"{MODEL}_BEST.txt", "w", newline='')
writer = csv.writer(f, delimiter="\t")

model_config = config(
    task="classification",
    metrics=["accuracy"],
    seed=42,
    **args
)
best = get_auc_over_k_folds(model_config)
f.write(f"best = {best}\n")

f.close()