"""
Code for the encoder of TransPoseNet
 code is based on https://github.com/facebookresearch/detr/tree/master/models
 (transformer + position encoding. Note: LN at the end of the encoder is not removed)
 with the following modifications:
- decoder is removed
- encoder is changed to take the encoding of the pose token and to output just the token
"""

import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention import MultiHeadAttention


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                pos: Optional[Tensor] = None):
        output = src

        for i, layer in enumerate(self.layers):
            output, res_mean = layer(output, pos=pos)
            if i == 0:
                res_means = res_mean.unsqueeze(1) # (B, 1, H)
            else:
                res_means = torch.cat([res_means, \
                                          res_mean.unsqueeze(1)], dim=1) # (B, L, H)
        if self.norm is not None:
            output = self.norm(output)

        return output, res_means


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.nhead = nhead
        self.head_dim = d_model//nhead

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, _, q, k = self.self_attn(q, k, value=src2) # (B*H, N, C//H)
        q_mean = q.mean(dim=1).view(-1, self.nhead, self.head_dim) # (B*H, C//H) -> (B, H, C//H)
        k_mean = k.mean(dim=1).view(-1, self.nhead, self.head_dim) # (B*H, C//H) -> (B, H, C//H)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, torch.norm(q_mean-k_mean, p=2, dim=-1)

    def forward(self, src, pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



