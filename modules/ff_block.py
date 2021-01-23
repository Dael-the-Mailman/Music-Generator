"""
Wrapper for feed-forward block

Used the implementation in https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
"""
import torch
import torch.nn as nn

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size=360, expansion=4, drop_p=0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )