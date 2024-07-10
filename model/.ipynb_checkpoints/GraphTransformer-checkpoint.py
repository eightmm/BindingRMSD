import torch, dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.glob import AvgPooling, SumPooling

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_input_feats, num_output_feats, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads

        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)
        self.E = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads)

    def propagate_attention(self, g):
        g.apply_edges(lambda edges: {"score":  edges.src['K_h'] * edges.dst['Q_h']})  ## multiply
        g.apply_edges(lambda edges: {"score": (edges.data["score"] / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)})  ## scale & clip
        g.apply_edges(lambda edges: {"score":  edges.data['score'] * edges.data['proj_e']})        ## dot production
        g.apply_edges(lambda edges: {"e_out":  edges.data["score"]})
        g.apply_edges(lambda edges: {"score":  torch.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0, 5.0))})   ## softmax & clip
        g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))   ## dot production & sum
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_e = self.E(edge_feats)
            g.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
            g.edata['proj_e'] = edge_feats_e.view(-1, self.num_heads, self.num_output_feats)

            self.propagate_attention(g)

            h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-8))  # Add eps to all
            e_out = g.edata['e_out']

        return h_out, e_out

class GraphTransformerModule(nn.Module):
    def __init__(self, num_hidden_channels, residual=True, num_attention_heads=4, dropout_rate=0.1):
        super(GraphTransformerModule, self).__init__()
        self.activ_fn = nn.SiLU()
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels

        self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer( self.num_hidden_channels, self.num_output_feats // self.num_attention_heads, self.num_attention_heads )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

    def forward(self, g, node_feats, edge_feats):
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        node_feats = self.layer_norm1_node_feats(node_feats)
        edge_feats = self.layer_norm1_edge_feats(edge_feats)

        node_attn_out, edge_attn_out = self.mha_module(g, node_feats, edge_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        node_feats = self.layer_norm2_node_feats(node_feats)
        edge_feats = self.layer_norm2_edge_feats(edge_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        return node_feats, edge_feats
