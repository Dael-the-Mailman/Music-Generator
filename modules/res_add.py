"""
Wrapper for residual addition

Used the implementation in https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
"""
import torch
import torch.nn as nn

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x