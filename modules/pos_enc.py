"""
Encodes the input spectrogram tensor using the algorithm outlined in
https://arxiv.org/pdf/1706.03762.pdf

Code is modified from https://github.com/wzlxjtu/PositionalEncoding2D

NOTE: Intended shape (song_duration, 1846, embed_size)
"""
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, dropout=0.5, depth=2, height=201, width=221):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pass