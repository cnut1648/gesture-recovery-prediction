"""
Use XGBosst to predict clinical only data (tabular)
"""

from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score
import optuna, os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_tabular.models import TabNetModel, TabNetModelConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from pytorch_tabular.models.autoint import AutoIntConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

# MODEL = "tabtransformer"
# MODEL = "FTtransformer"
MODEL = "autoint"

data_dir = Path("/home/jiashu/seq/raw_data")
processed_data_dir = Path("/home/jiashu/seq/processed/train30_test10")

df = pd.read_excel(
    data_dir / "Dissection Gesture_Clinical Data.xlsx",
    engine='openpyxl'
)

df['Nerve Sparing'] = df['Nerve Sparing'].apply(
    lambda val: 1 if val == 'full' else 0
)
X = df.drop(["Case ID", "Surgeon ID", "Pre-Op SHIM",], axis=1)

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

# model_config = TabNetModelConfig(
#     task="classification",
#     metrics=["accuracy"],
#     learning_rate=1e-3,
# )
# model_config = CategoryEmbeddingModelConfig(
#     task="classification",
#     layers="1024-512-512",  # Number of nodes in each layer
#     activation="LeakyReLU", # Activation between each layers
#     learning_rate = 1e-3,
#     metrics=["accuracy"],
#     metrics_params=[{}]
# )


def train_model(model_config, k_fold) -> float:
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    k_dir = processed_data_dir / f"k{k_fold}"
    with open(k_dir / "train.pkl", "rb") as f:
        seq = pickle.load(f)
    train_idxs = df['Case ID'].isin(seq.keys()).values
    X_train = X[train_idxs]

    with open(k_dir / "valid.pkl", "rb") as f:
        seq = pickle.load(f)
    test_idxs = df['Case ID'].isin(seq.keys()).values
    X_test = X[test_idxs]

    tabular_model.fit(train=X_train, validation=X_test)
    pred_df = tabular_model.predict(X_test)
    prob = pred_df['1_probability']
    try:
        return roc_auc_score(X_test['ESI @ 12 mo'], prob)
    except:
        return 0.0

def get_auc_over_k_folds(model_config):
    rets = []
    for k_fold in range(4):
        auc = train_model(model_config, k_fold)
        rets.append(auc)
    return np.mean(rets)

def objective_tabtransformer(trial):
    emb_dim = trial.suggest_categorical("input_embed_dim", [16, 32, 64, 128, 256])
    emb_dropout = trial.suggest_float("embedding_dropout", 0.1, 0.6)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
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
study.optimize(models_objectives[MODEL], n_trials=3000)


trial = study.best_trial

print(f"best study has avg acc over k-fold = {trial.value}")
for key, value in trial.params.items():
  print("    {}: {}".format(key, value))
