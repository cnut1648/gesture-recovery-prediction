from typing import List
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from functools import partial
import copy


class IMVTensorLSTM(nn.Module):
    
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.U_j.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.U_j.device)
        outputs: List[Tensor] = []
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean, alphas, betas

   
class TwoHeadIMVLSTM(nn.Module):
    def __init__(self,
        # emb_dim also RNN input dim
        num_emb, emb_dim = 128,
        # RNN
        hidden_dim = 1024, n_units = 128,
        linear_dim = 512,
        **kwargs):

        super(TwoHeadIMVLSTM, self).__init__()

        # need to have separate layers for IG
        self.embeddings_L = nn.Embedding(num_emb, emb_dim)
        self.embeddings_R = copy.deepcopy(self.embeddings_L)
        # self.embeddings_R = nn.Embedding(num_emb, emb_dim)

        # for occlusion zero emb
        # self.register_buffer("zero_emb", torch.zeros(emb_dim))
        self.zero_emb =  torch.zeros(emb_dim)
        

        self.lstm = IMVTensorLSTM(
            emb_dim, hidden_dim, n_units
        )
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.out = nn.Linear(linear_dim, 1)

        self.relu = nn.ReLU()
        
    def forward(self, L: dict, R: dict, L_emb=None, R_emb=None, **kwargs):
        if L_emb is not None and R_emb is not None:
            emb_gesture_L = L_emb
            emb_gesture_R = R_emb
        else:
            gesture_L: PackedSequence = L["gesture"]

            def emb(packed_sequence: PackedSequence):
                embs = []
                for gesture in packed_sequence.data.unbind():
                    # occlusion, use -1 for zero emb
                    if gesture.item() == -1:
                        embs.append(self.zero_emb)
                    else:
                        embs.append(self.embeddings_L(gesture))
                embs = torch.stack(embs, dim=0)
                return PackedSequence(embs, packed_sequence.batch_sizes)

            emb_gesture_L: PackedSequence = emb(gesture_L)
            # pad_L (bsz, T, d=emb_dim), len_L (bsz, )
            pad_L, len_L = pad_packed_sequence(emb_gesture_L, batch_first=True)

            gesture_R: PackedSequence = R["gesture"]
            emb_gesture_R: PackedSequence = emb(gesture_R)
            pad_R, len_R = pad_packed_sequence(emb_gesture_R, batch_first=True)

        # (bsz, hidden_dim)
        h_L, _, _ = self.lstm(pad_L)

        # (bsz, hidden_dim)
        h_R, _, _ = self.lstm(pad_R)

        # (N, 2 * hidden_dim)
        h = torch.cat([h_L, h_R], dim=1)
        h = self.relu(self.linear(h))
        # (N, 1)
        out = self.out(h)
        return out.view(-1)