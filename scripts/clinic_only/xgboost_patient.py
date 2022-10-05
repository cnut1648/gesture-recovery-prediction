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

def acc_for_k_fold(xgb, X, y, k_fold: int, plot=False, use_auc=True):
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
    
    xgb.fit(
        X_train, y_train,
        verbose=0,
        eval_set=[(X_test, y_test)],
        eval_metric=["auc"]
    )
    if use_auc:
        # auc
        prob = xgb.predict_proba(X_test)
        ret = roc_auc_score(y_test, prob[:, 1])
    else:
        # acc
        ret = xgb.score(X_test, y_test)
    if plot:
        img_folder = "/home/jiashu/seq/artifact/xgboost"
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)
        feature_names = X.columns
        sorted_idxs = xgb.feature_importances_.argsort()
        print(xgb.feature_importances_)
        plt.figure()
        plt.barh(
            feature_names[sorted_idxs],
            xgb.feature_importances_[sorted_idxs]
        )
        metric = "AUC" if use_auc else "accuracy"
        plt.title(f"k={k_fold}, test {metric} = {ret}")
        plt.savefig(f"{img_folder}/k{k_fold}.png", bbox_inches="tight")
        xgb.save_model(f"/home/jiashu/seq/selected_ckpt/xgboost/client-k{k_fold}.json")

    return ret


def objective(trial):
    # order of important

    # number of grad boosted tree = #boosting rounds
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)

    # max depth of tree
    # larger -> more complex model -> overfit
    max_depth = trial.suggest_int("max_depth", 3, 500)

    # L1 regulartion on weights
    # larger -> model more conservative -> underfit
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)

    # minimum sum of weights of all observations required in a child
    # high lead to less child, prevent model from learn relation specific
    #   to input sample only
    # higher -> conservative -> underfit
    min_child_weight = trial.suggest_float("min_child_weight", 0, 1)

    # In maximum delta step we allow each tree’s weight estimation to be
    # 0 = no constraint; positive = update step more conservative
    # useful in logistic regression under imbalanced data
    max_delta_step = trial.suggest_float("max_delta_step", 0, 10)

    # fraction of observations to be randomly samples for each tree
    # Subsample ratio of the training instances
    # 0.5 = random sample half of training data 
    #   prior to growing trees -> prevent overfit
    # once in every boosting iteration
    # lower -> conservative -> prevent overfit
    subsample = trial.suggest_float("subsample", 0.5, 1)

    # eta, step size shrinkage used in update to prevent overfitting
    # After each boosting step, 
    #   directly get the weights of new features, 
    #   and eta shrinks the feature weights 
    #   to make the boosting process more conservative.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    # A node is split only when the resulting split gives a positive reduction in the loss function
    # specifies the minimum loss reduction required to make a split
    # larger -> model more conservative -> underfit
    gamma = trial.suggest_float("gamma", 0, 5)

    # L2 regulartion on weights
    # larger -> model more conservative -> underfit
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 3)

    # # Subsample ratio of columns when constructing each tree.
    # colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
    # # Subsample ratio of columns when constructing each level.
    # colsample_bylevel = trial.suggest_float('colsample_bylevel', 0, 1)
    # # Subsample ratio of columns when constructing each split.
    # colsample_bynode = trial.suggest_float('colsample_bynode', 0, 1)

    rets = []
    for k_fold in range(4):
        xgb = XGBClassifier(
            objective="binary:logistic",
            max_depth=max_depth,
            n_estimators=n_estimators,
            use_label_encoder=False,
            verbosity=0,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step
        )
        ret = acc_for_k_fold(
            xgb, X, y, k_fold=k_fold)
        rets.append(ret)
    return np.mean(rets)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)

trial = study.best_trial

print(f"best study has avg acc over k-fold = {trial.value}")
for key, value in trial.params.items():
  print("    {}: {}".format(key, value))

# save feature importance
for k_fold in range(4):
    xgb = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        verbosity=0,
        **trial.params
    )
    acc_for_k_fold(xgb, X, y, k_fold=k_fold, plot=True)