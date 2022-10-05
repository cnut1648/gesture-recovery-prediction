import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModel(nn.Module):
    """
    token 19 orig + 3 (special token)
    """

    def __init__(self, d_model: int, n_encoder_layers=1, dropout = 0.3, n_head=8, hidden_dim=1024, **kwargs):
        super(AttnModel, self).__init__()
        # self.emb_L = nn.Embedding(19 + 3, d_model)
        # self.pe_L = PositionalEncoding(d_model, 0.3)
        # self.encoder_L = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model, n_head, hidden_dim, dropout=0.3),
        #     num_layers=n_encoder_layers
        # )

        # self.emb_R = nn.Embedding(19 + 3, d_model)
        # self.pe_R = PositionalEncoding(d_model, 0.3)
        # self.encoder_R = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model, n_head, hidden_dim, dropout=0.3),
        #     num_layers=n_encoder_layers
        # )
        self.emb = nn.Embedding(19 + 3, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_head, hidden_dim, 
                dropout=dropout),
            num_layers=n_encoder_layers
        )

        self.projection = nn.Linear(d_model, 1)

    def forward(self, 
        # (bsz, L), (bsz, L)
        input_ids_L, attn_mask_L, 
        input_ids_R, attn_mask_R, 
        # used in combined model
        output_emb=False, 
        # used in occlusion, 1 if valid, 0 if occluded
        # (bsz, L)
        occlusion_idx_L=None, occlusion_idx_R=None,
        **unused):

        emb_L = self.emb(input_ids_L)
        emb_L = self.pe(emb_L)
        if occlusion_idx_L is not None:
            emb_L = occlusion_idx_L.unsqueeze(-1) * emb_L
        # (bsz L d) -> (L bsz d)
        L = self.encoder(emb_L.permute(1, 0, 2), src_key_padding_mask=(~attn_mask_L))
        # use [CLS]
        L = L[0]

        emb_R = self.emb(input_ids_R)
        emb_R = self.pe(emb_R)
        if occlusion_idx_R is not None:
            emb_R = occlusion_idx_R.unsqueeze(-1) * emb_R
        R = self.encoder(emb_R.permute(1, 0, 2), src_key_padding_mask=(~attn_mask_R))
        # use [CLS]
        # R = R[:, 0, :]
        R = R[0]

        LR = (L + R) / 2
        if output_emb:
            return LR
        out = self.projection(LR)
        return out.view(-1)
    
    def get_scores(self, 
            input_ids_L, attn_mask_L, 
            input_ids_R, attn_mask_R, 
            **unused):
        scores = []
        def hook(module, input, output):
            _, score = output
            scores.append(score)

        for layer in self.encoder.layers:
            layer.self_attn.register_forward_hook(hook)
        # assert 0 <= layer_idx < self.encoder.num_layers
        layer = self.encoder.layers[0]
        emb_L = self.emb(input_ids_L)
        emb_L = self.pe(emb_L)
        # one forward
        self.encoder(emb_L.permute(1, 0, 2), src_key_padding_mask=(~attn_mask_L))

        scores_L = (
            # first layer
            scores[0],
            # take avg
            # (bsz, #layer, L, L) -> (bsz, L, L)
            torch.stack(scores, dim=1).mean(1)
        )

        scores.clear()
        emb_R = self.emb(input_ids_R)
        emb_R = self.pe(emb_R)
        # another forward
        self.encoder(emb_R.permute(1, 0, 2), src_key_padding_mask=(~attn_mask_R))

        scores_R = (
            # first layer
            scores[0],
            # take avg
            # (bsz, #layer, L, L) -> (bsz, L, L)
            torch.stack(scores, dim=1).mean(1)
        )

        return scores_L, scores_R

