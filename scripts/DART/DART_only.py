import numpy as np
import optuna
import pandas as pd

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score

eval = True

def load_DART(kfold, split):
    X = pd.read_csv(f"/home/jiashu/seq/processed/train30_test10/k{kfold}/{split}-DART.csv")
    y = X['label'].tolist()
    X = X.drop(["case", "label"], axis=1)
    return X, y

def acc_for_k_fold(xgb, kfold, save=False):
    X_train, y_train = load_DART(kfold, 'train')
    X_val, y_val = load_DART(kfold, 'valid')

    xgb.fit(
        X_train, y_train,
        verbose=0,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"]
    )
    prob = xgb.predict_proba(X_val)
    if save:
        xgb.save_model(f"/home/jiashu/seq/selected_ckpt/DART/DART_only/k{kfold}.json")
    return roc_auc_score(y_val, prob[:, 1])


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
    for kfold in range(4):
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
            xgb, kfold)
        rets.append(ret)
    
    aucs = np.array(rets)
    aucs = np.where(aucs > 0.5, aucs, 1 - aucs)
    # print(aucs, aucs.mean(), aucs.std(ddof=1))
    return aucs.mean()

if not eval:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)

    trial = study.best_trial

    print(f"best study has avg acc over k-fold = {trial.value}")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save feature importance
    for kfold in range(4):
        xgb = XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            verbosity=0,
            **trial.params
        )
        acc_for_k_fold(xgb, kfold, save=True)
else:
    rets = []
    for kfold in range(4):
        xgb = XGBClassifier(
            objective="binary:logistic",
        )
        xgb.load_model(f"/home/jiashu/seq/selected_ckpt/DART/DART_only/k{kfold}.json")
        ret = acc_for_k_fold(xgb, kfold)
        rets.append(ret)
    aucs = np.array(rets)
    aucs = np.where(aucs > 0.5, aucs, 1 - aucs)
    print(aucs, aucs.mean(), aucs.std(ddof=1))
    
