"""
Shared attention modules for set-based feature extractors and actor-critics.

Based on the Set Transformer paper (Lee et al., 2019):
https://github.com/juho-lee/set_transformer/blob/6fdae7f/modules.py

Uses PyTorch's native scaled_dot_product_attention for efficiency.
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block.
    
    Computes multi-head cross-attention between query Q and key-value K,
    with residual connections, optional layer normalization, and a 
    feed-forward network.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        bs, q_len = Q.size(0), Q.size(1)
        k_len = K.size(1)

        Q_proj = self.fc_q(Q)
        K_proj, V_proj = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads

        # Reshape to 4D tensors for native SDPA: [bs, num_heads, seq_len, dim_split]
        Q_ = Q_proj.view(bs, q_len, self.num_heads, dim_split).transpose(1, 2)
        K_ = K_proj.view(bs, k_len, self.num_heads, dim_split).transpose(1, 2)
        V_ = V_proj.view(bs, k_len, self.num_heads, dim_split).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q_, K_, V_, scale=1.0 / math.sqrt(self.dim_V)
        )
        
        # Reshape back to 3D and apply residual connection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, q_len, self.dim_V)
        O = Q_proj + attn_output

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Set Attention Block.
    
    Self-attention on a set X: SAB(X) = MAB(X, X).
    Captures pairwise interactions between elements of an unordered set.
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Set Attention Block.
    
    Efficient set attention using M learnable inducing points I to compute
    attention in O(N*M) time instead of O(N^2).
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(th.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.expand(X.size(0), -1, -1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention.
    
    Aggregates set representations into k fixed seed vectors S via MAB(S, X).
    Used with num_seeds=1 to pool unordered sets into a single embedding.
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(th.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)


class MeanPool(nn.Module):
    """Mean pooling over the sequence dimension.
    
    Lightweight drop-in replacement for PMA when the input size is fixed.
    """
    def __init__(self, dim, ln=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim) if ln else nn.Identity()

    def forward(self, X):
        # X: [bs, n, dim] -> [bs, 1, dim]
        return self.norm(self.proj(X.mean(dim=1, keepdim=True)))
