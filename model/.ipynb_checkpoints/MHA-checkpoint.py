import torch, math
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.head_dim = embed_dim // num_heads

        # Linear transformations for query, key, and value projections
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

        # Linear transformation for the output of multi-head attention
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, h):
        # Project query, key, and value using linear transformations
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Split the queries, keys, and values into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply softmax to compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Merge the heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1)

        # Project the output
        output = self.out_proj(attn_output)

        # Dropout
        output = F.dropout(h, 0.1, training=self.training)

        return output, attn_weights
