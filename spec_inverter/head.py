"""
Implements a modified convolutional head as described in
https://arxiv.org/pdf/1808.06719.pdf
"""
import torch
import torch.nn as nn

class Head(nn.Sequential):
    def __init__(self, channels=2):
        pass
    
    def conv_layer(self, in_size, out_size, width, stride, filters):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, width, stride),
            nn.ELU()
        )