"""
Encodes the input spectrogram tensor using the algorithm outlined in
https://arxiv.org/pdf/1706.03762.pdf

Code is modified from https://github.com/wzlxjtu/PositionalEncoding2D

NOTE: Intended shape (2, 201, 221, 360)
"""
import torch
import pytorch_lightning as pl

class PosEnc(pl.LightningModule):
    def __init__(self, d_model, dropout=0.5, depth=2, height=201, width=221):
        super().__init__()

    def forward(self, x):
        pass