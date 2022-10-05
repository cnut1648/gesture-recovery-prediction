import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from functools import partial
import copy

def elementwise_apply(fn, packed_sequence: PackedSequence):
    """
    applies a pointwise function fn to each element in packed_sequence
    https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184
    """
    return PackedSequence(packed_sequence.data.apply_(fn), packed_sequence.batch_sizes)

class TwoHeadLSTM(nn.Module):
    def __init__(self,
        # emb_dim also RNN input dim
        num_emb = 25, emb_dim = 128,
        # RNN
        hidden_dim = 1024, num_layers = 2,
        # output MLP -> 1 (binary clf)
        linear_dim = 1024,
        use_same_lstm=True,
        dropout=0.0, **kwargs):

        if num_layers == 1:
            dropout = 0.0
        super(TwoHeadLSTM, self).__init__()

        # need to have separate layers for IG
        self.embeddings_L = nn.Embedding(num_emb, emb_dim)
        self.embeddings_R = copy.deepcopy(self.embeddings_L)

        # for occlusion zero emb
        # self.register_buffer("zero_emb", torch.zeros(emb_dim))
        self.zero_emb =  torch.zeros(emb_dim)
        

        self.lstm = nn.LSTM(emb_dim,
                            hidden_dim, num_layers,
                            batch_first=False, dropout=dropout)
        # if use_same_lstm == True:
        #     self.lstm_R = self.lstm
        # else:
        #     self.lstm_R = nn.LSTM(emb_dim,
        #                     hidden_dim, num_layers,
        #                     batch_first=False, dropout=dropout)

        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.out = nn.Linear(linear_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
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

            gesture_R: PackedSequence = R["gesture"]
            emb_gesture_R: PackedSequence = emb(gesture_R)

        _, (h_L, _) = self.lstm(emb_gesture_L)
        # last layer hidden state (N, hidden_dim)
        h_L = h_L[-1]

        # _, (h_R, _) = self.lstm_R(emb_gesture_R)
        _, (h_R, _) = self.lstm(emb_gesture_R)
        # last layer hidden state (N, hidden_dim)
        h_R = h_R[-1]

        # (N, 2 * hidden_dim)
        h = torch.cat([h_L, h_R], dim=1)
        h = self.relu(self.linear(h))
        # (N, 1)
        out = self.out(h)
        return out.view(-1)