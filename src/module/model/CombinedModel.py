from typing import Optional
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from torch import nn
from pytorch_tabular import TabularModel

from src.module.BaselineModule import BaselineModule

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


class CombinedModel(nn.Module):
    def __init__(self,
        gesture_dim: int, clinical_dim: int,
        gesture_ckpt: str, clinical_ckpt: str,
        # tabular_args, gesture_args,
        mlp_args, hidden_dim, n_hidden_layers,
        **kwargs
    ):
        super().__init__()
        assert gesture_dim >= clinical_dim 
        # align clinical space to gesture space
        self.space_aligner = nn.Linear(clinical_dim, gesture_dim)
        # learn importance of gesture model
        self.alpha = nn.Sequential(
            nn.Linear(gesture_dim, 1),
            nn.Sigmoid()
        )
        self.mlp = MLP_factory(**mlp_args, layer_sizes=[
            [gesture_dim, 1],
            [hidden_dim, n_hidden_layers],
            [1, 1]
        ])

        # self.clinic_model = TabularModel.load_from_checkpoint(clinical_ckpt)
        # self.clinic_model = DeepFeatureExtractor(self.clinic_model)

        self.gesture_model = BaselineModule.load_from_checkpoint(gesture_ckpt).model
        self.gesture_model.train()
        for param in self.gesture_model.parameters():
            param.requires_grad = True
    
    def forward(self, 
        # (bsz, clinical_dim)
        clinical,
        # (bsz, L), (bsz, L)
        input_ids_L, attn_mask_L, 
        input_ids_R, attn_mask_R, 
        # used in occlusion, 1 if valid, 0 if occluded
        # (bsz, L)
        occlusion_idx_L=None, occlusion_idx_R=None,
        **unused):
        # clinical_emb = self.clinic_model.transform(clinical.detach().cpu().numpy()).values
        gesture_emb = self.gesture_model(
            input_ids_L, attn_mask_L, 
            input_ids_R, attn_mask_R, 
            output_emb=True
        )
        clinical2gesture = self.space_aligner(clinical)
        alpha = self.alpha(gesture_emb)
        combined = alpha * gesture_emb * (1 - alpha) * clinical2gesture
        return self.mlp(combined).view(-1)
