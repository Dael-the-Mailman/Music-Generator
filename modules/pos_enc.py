"""
Encodes the input spectrogram tensor using the algorithm outlined in
https://arxiv.org/pdf/1706.03762.pdf

Code is modified from https://github.com/wzlxjtu/PositionalEncoding2D

NOTE: Intended shape (song_duration, 1846, embed_size)
"""
import math
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, embed_size=360, dropout=0.5, depth=2, height=201, width=221):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, x):
        width = x.shape[0]
        height = x.shape[1]
        pe = torch.zeros(width, height, self.embed_size)
        dim = self.embed_size//2
        div_term = torch.exp(torch.arange(0., dim, 2) *
                             -(math.log(10_000.0) / dim))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[:, :, 0:dim:2] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, height, 1)
        pe[:, :, 1:dim:2] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, height, 1)
        pe[:, :, dim::2] = torch.sin(pos_h * div_term).unsqueeze(0).repeat(width, 1, 1)
        pe[:, :, dim + 1::2] = torch.cos(pos_h * div_term).unsqueeze(0).repeat(width, 1, 1)

        return self.dropout(x + pe)