"""
Encodes the input spectrogram tensor using the algorithm outlined in
https://arxiv.org/pdf/1706.03762.pdf

Code is modified from https://github.com/wzlxjtu/PositionalEncoding2D

NOTE: Intended shape (song_duration, 1846, 360)
"""
import torch
import torch.nn

class PosEnc(nn.Module):
    def __init__(self, d_model, dropout=0.5, depth=2, height=201, width=221):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(depth, height, width, d_model)

    def forward(self, x):
        pass