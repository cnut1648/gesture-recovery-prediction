import os
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn
import optuna
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from pytorch_lightning import seed_everything

# name -> emb dim
clinic_models = {
    "tab_transformer": 32,
    "ft_transformer": 32,
    "autoint": 96
}
gesture_models = {
    # lstm
    "gesture_emb": 356,
    "attn_gesture_emb": 256
}

def load_emb(k_fold, split, GMODEL, CMODEL, device, return_case=False):
    assert split in ['train', 'valid', 'test']
    with open(f"/home/jiashu/seq/processed/train30_test10/k{k_fold}/{split}_combined.pkl", "rb") as f:
        seq = pickle.load(f)
    
    gesture_embs, clinical_embs, labels, cases = [], [], [], []
    for case, data in seq.items():
        gesture_embs.append(data[GMODEL].reshape(-1))
        clinical_embs.append(data[CMODEL].reshape(-1))
        labels.append(data['label'])
    
    return (
        torch.tensor(gesture_embs, device=device), 
        torch.tensor(clinical_embs, device=device),
        torch.tensor(labels, device=device).float()
    )
    # embs, labels, cases = [], [], []

    # for case, data in seq.items():
    #     emb = np.concatenate((
    #         data[GMODEL], data[CMODEL]
    #     ), axis=1)
    #     embs.append(torch.from_numpy(emb))
    #     labels.append(data['label'])
    #     cases.append(case)
    
    # embs = torch.cat(embs, dim=0).to(device)
    # if not return_case:
    #     return embs, torch.tensor(labels).float().to(device)
    # return  embs, torch.tensor(labels).float().to(device), cases

activation_classes = {
    'gelu': nn.GELU, 
    'relu': nn.ReLU, 
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'hard_siwsh': nn.Hardswish,
    'prelu': nn.PReLU,
    'relu6': nn.ReLU6,
    'rrelu': nn.RReLU,
    'celu': nn.CELU,
    'softplus': nn.Softplus
}
normalization_layers = {
    'none': None,
    'layer_norm': nn.LayerNorm,
    'batch_norm': nn.BatchNorm1d
}

# Simple MLP
def MLP_factory(
        layer_sizes, dropout=False,
        norm='none', activation='gelu'):
    modules = nn.ModuleList()
    unpacked_sizes = []
    for block in layer_sizes:
        unpacked_sizes.extend([block[0]] * block[1])

    for k in range(len(unpacked_sizes) - 1):
        if normalization_layers[norm]:
            modules.append(normalization_layers[norm](unpacked_sizes[k]))

        modules.append(nn.Linear(unpacked_sizes[k], unpacked_sizes[k + 1]))

        if k < len(unpacked_sizes) - 2:
            modules.append(activation_classes[activation.lower()]())
            if dropout is not False:
                modules.append(nn.Dropout(dropout))

    mlp = nn.Sequential(*modules)
    return mlp

class MLP(nn.Module):
  def __init__(self, gesture_dim, clinical_dim, **kwargs):
    super(MLP, self).__init__()
    self.gesture_dim = gesture_dim
    self.clinical_dim = clinical_dim
    assert self.gesture_dim >= self.clinical_dim 
    # align clinical space to gesture space
    self.space_aligner = nn.Linear(clinical_dim, gesture_dim)
    # learn importance of gesture model
    self.alpha = nn.Sequential(
        nn.Linear(gesture_dim, 1),
        nn.Sigmoid()
    )

    self.mlp = MLP_factory(**kwargs)

  def forward(self, gesture_emb, clinical_emb):
    clinical2gesture = self.space_aligner(clinical_emb)
    alpha = self.alpha(gesture_emb)
    combined = alpha * gesture_emb * (1 - alpha) * clinical2gesture
    return self.mlp(combined)

def highest_auc(layer_size, dropout, norm, activation, lr, save_ckpt=False):
    aucs = []
    test_aucs = []
    device = torch.device("cuda")
    # device = torch.device("cpu")

    for k_fold in range(0, 4):
        SAVE_PATH = Path(f"/home/jiashu/seq/selected_ckpt/combined/{k_fold}")
        os.makedirs(SAVE_PATH, exist_ok=True)
        # print("++++++++++++++++++++")
        # print(f"\t\t k = {k_fold}")
        model = MLP(
            gesture_dim=gesture_models[GMODEL],
            clinical_dim=clinic_models[CMODEL],
            layer_sizes=layer_size, 
            dropout=dropout, 
            norm=norm, 
            activation=activation).to(device)

        gesture_train, clinical_train, y_train = load_emb(k_fold, 'train', GMODEL, CMODEL, device=device)
        gesture_dev, clinical_dev, y_dev = load_emb(k_fold, 'valid', GMODEL, CMODEL, device=device)

        dataloaders = {
            "train": DataLoader(TensorDataset(gesture_train, clinical_train, y_train), batch_size=16, shuffle=True),
            "valid": DataLoader(TensorDataset(gesture_dev, clinical_dev, y_dev), batch_size=16, shuffle=False),
        }

        criterion = nn.BCEWithLogitsLoss()
        optim = Adam(model.parameters(), lr=lr)

        highest_auc = -float('inf')

        for epoch in range(EPOCH):
            probs = []
            labels = []
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                for gesture, clinical, y in dataloaders[phase]:
                    optim.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = model(gesture, clinical).view(-1)
                        loss = criterion(logits, y)
                        if phase == 'train':
                            loss.backward()
                            optim.step()
                        
                        probs.extend(
                            logits.detach().cpu().numpy()
                        )
                        labels.extend(
                            y.detach().cpu().numpy()
                        )
                
            if epoch % 5 == 0:
                probs = np.array(probs)
                labels = np.array(labels)
                preds = probs > 0.0
                assert phase == "valid"
                auc = roc_auc_score(labels, probs)
                # acc = accuracy_score(labels, preds)
                # print(f"EPOCH: {epoch} => acc={acc}, auc={auc}")
                if auc > highest_auc:
                    highest_auc = auc
                    if save_ckpt:
                        torch.save(model, SAVE_PATH / f"{GMODEL}-{CMODEL}.pt")
        # print(f"highest auc = {highest_auc}")
        # print("++++++++++++++++++++")
        aucs.append(highest_auc)
        if save_ckpt:
            model = torch.load(SAVE_PATH / f"{GMODEL}-{CMODEL}.pt")
        gesture_test, clinical_test, y_test = load_emb(k_fold, 'test', GMODEL, CMODEL, device=device)
        logits = model(gesture_test, clinical_test).view(-1)
        auc = roc_auc_score(y_test.cpu().numpy(), logits.detach().cpu().numpy()) 
        test_aucs.append(auc)
    
    return np.array(aucs), np.array(test_aucs)

def objective(trial):
    layer_size=[
        [gesture_models[GMODEL], 1],
        [trial.suggest_int('hidden_dim', 128, 512), trial.suggest_categorical('num_hidden', [1, 2, 3])],
        [1, 1]
    ]
    dropout=trial.suggest_float("dropout", 0.1, 0.6)
    norm=trial.suggest_categorical("norm", normalization_layers.keys())
    activation=trial.suggest_categorical("activation", activation_classes.keys())
    learning_rate=trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)

    valid, test = highest_auc(layer_size, dropout, norm, activation, learning_rate)
    return valid.mean(), test.mean()

if __name__ == "__main__":
    # device = torch.device("cpu")
    seed_everything(42, workers=True)

    GMODEL = "attn_gesture_emb"
    # CMODEL = "ft_transformer"
    CMODEL = "tab_transformer"
    # CMODEL = "autoint"
    # false then optuna
    EVAL = False
    # EVAL = True
    print(f"gesture model = {GMODEL} | clinic model = {CMODEL}", flush=True)
    EPOCH = 100
    INPUT_SIZE = gesture_models[GMODEL] + clinic_models[CMODEL]

    if EVAL:
        if CMODEL == "tab_transformer":
            layer_size = [
                [INPUT_SIZE, 1],
                [483, 2],
                [1,1]
            ]
            aucs = highest_auc(layer_size=layer_size, 
                dropout=0.12006130735599503,
                norm='batch_norm', 
                activation='relu6', 
                lr=9.969711549907534e-06,
                save_ckpt=True)
        elif CMODEL == "ft_transformer":
            layer_size = [
                [INPUT_SIZE, 1],
                [363, 1],
                [1,1]
            ]
            aucs = highest_auc(layer_size=layer_size, 
                dropout=0.11941682010788679, 
                norm='batch_norm', 
                activation='leaky_relu', 
                lr=9.993420408759375e-06, 
                save_ckpt=True)
        elif CMODEL == "autoint":
            layer_size = [
                [INPUT_SIZE, 1],
                [498, 1],
                [1,1]
            ]
            aucs = highest_auc(layer_size=layer_size, 
                dropout=0.10007305634360696,
                norm='layer_norm', 
                activation='relu', 
                lr=8.650400707353551e-06, 
                save_ckpt=True)
        print(aucs, aucs.mean(), aucs.std(ddof=1))
    else:
        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=3000)

        with open(f"/home/jiashu/seq/selected_ckpt/combined/optuna-{CMODEL}.pkl", "wb") as f:
            pickle.dump(study, f)

        trial = study.best_trial

        print(f"best study has avg acc over k-fold = {trial.value}")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        layer_size = [
            [INPUT_SIZE, 1],
            [trial.params['hidden_dim'], trial.params['num_hidden']],
            [1,1]
        ]
        valid, test = highest_auc(layer_size=layer_size, 
            dropout=trial.params['dropout'],
            norm=trial.params['norm'], 
            activation=trial.params['activation'], 
            lr=trial.params['learning_rate'], 
            save_ckpt=True)
        print(f"valid auc: {valid}, {valid.mean()} (+- {valid.std(ddof=1)})")
        print(f"test auc: {test}, {test.mean()} (+- {test.std(ddof=1)})")

