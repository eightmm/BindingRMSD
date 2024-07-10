import dgl, torch
import torch.nn as nn
import torch.nn.functional as F

from .GatedGCNLSPE import GatedGCNLSPELayer
from .MHA import MultiHeadAttention

class GraphGPS(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(GraphGPS, self).__init__()

        self.mpnn_layer = GatedGCNLSPELayer(
            emb_dim,
            emb_dim,
            dropout=0.1,
            batch_norm=True,
            use_lapeig_loss=False,
            residual=True
        )

        self.mha_layer  = MultiHeadAttention(
            emb_dim,
            num_heads
        )

        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )

        self.mpnn_bn = nn.BatchNorm1d(emb_dim)
        self.mha_bn  = nn.BatchNorm1d(emb_dim)
        self.mlp_bn  = nn.BatchNorm1d(emb_dim)

        self.mpnn_weight = nn.Parameter( torch.tensor( [1.0] ) )
        self.mha_weight  = nn.Parameter( torch.tensor( [1.0] ) )

    def forward(self, g, h, p, e):
        h_i = h
        e_i = e
        p_i = p

        h_mpnn, p, e   = self.mpnn_layer(g, h, p, e)
        h_mha, h_wight = self.mha_layer(h)

        h_mpnn += h_i
        h_mha  += h_i

        h_mpnn = self.mpnn_bn(h_mpnn)
        h_mha  = self.mha_bn(h_mha)

        h_j = h_mpnn * self.mpnn_weight + h_mha * self.mha_weight

        h = self.MLP(h_j)
        h += h_j

        h = self.mlp_bn(h)

        return h, p, e