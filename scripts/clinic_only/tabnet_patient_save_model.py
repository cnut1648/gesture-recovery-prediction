"""
Use XGBosst to predict clinical only data (tabular)
"""

from pytorch_tabular.models.autoint import AutoIntConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score
import optuna, os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_tabular.models import TabNetModel, TabNetModelConfig
from pytorch_tabular.models.tab_transformer import TabTransformerModel, TabTransformerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

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
    max_epochs=15,
    gpus=1, #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

def train_model(k_fold) -> float:
    # output = "tab_transformer"
    # args = {
    #     'input_embed_dim': 16, 
    #     'embedding_dropout': 0.15130781721503087, 
    #     'learning_rate': 0.0034269579096804378, 
    #     'share_embedding': True, 
    #     'out_ff_layers': '128-64-32', 
    #     'use_batch_norm': True
    # }
    # model_config = TabTransformerConfig(
    #     task="classification",
    #     metrics=["accuracy"],
    #     seed=42,
    #     **args
    # )
    # output = "ft_transformer"
    # args = {
    #     'input_embed_dim': 16, 
    #     'embedding_dropout': 0.5102183725345785, 
    #     'learning_rate': 6.674249922954897e-05, 
    #     'share_embedding': True, 
    #     "num_attn_blocks": 6,
    #     "attn_dropout": 0.4944460892125036
    # }
    # model_config = FTTransformerConfig(
    #     task="classification",
    #     metrics=["accuracy"],
    #     seed=42,
    #     **args
    # )
    output = "autoint"
    args = {
        'learning_rate': 6.476425269725806e-05, 
        'attn_embed_dim': 8, 
        "num_attn_blocks": 2,
        "attn_dropouts": 0.10804925361803194,
        'has_residuals': True, 
        'embedding_dim': 64, 
        'embedding_dropout': 0.12137643336994963, 
    }

    model_config = AutoIntConfig(
        task="classification",
        metrics=["accuracy"],
        seed=42,
        **args
    )
    
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
    tabular_model.save_model(f"/home/jiashu/seq/selected_ckpt/{output}/k{k_fold}")
    prob = pred_df['1_probability']
    return roc_auc_score(X_test['ESI @ 12 mo'], prob)
    
rets = []
for k_fold in range(4):
    auc = train_model(k_fold)
    rets.append(auc)
rets = np.array(rets)
rets = np.where(rets > 0.5, rets, 1-rets)
print(rets)
print( np.mean(rets), np.std(rets, ddof=1))