"""
Implementation of patch embedding

Code modified from:
https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

Input shape: (1, song_duration, 2, 201, 221)

:param depth: the number of audio channels
:patch_size: size of convolution and stride
:convpad: resize/pad the convolution so that it is evenly divisible
          by the patch_size
:embed_size: size of the embedding
:poolpad: resize/pad the pooling so that it is evenly divisible
          by the

TODO: make more robust position parameters
"""
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import repeat

class PatchEmbedding(nn.Module):
    def __init__(self, embed_size=360, depth=2, patch_size=15, convpad_h=12, convpad_w=2):
        super().__init__()
        # self.patch_size = patch_size
        # self.padding = padding
        self.embed_size = embed_size
        self.projection = nn.Sequential(
            nn.Conv2d(depth, embed_size, kernel_size=patch_size,
                                        # 201x221 -> 225x225 
                      stride=patch_size, padding=(convpad_h, convpad_w),
                      padding_mode='reflect'),
            # Output should be [batch_size, 41*45, embed_size]
            Rearrange('b e (h) (w) -> b (h w) e')
        )
    
        self.cls_tokens = nn.Parameter(torch.randn(1,1,embed_size))

    def forward(self, x):
        x = x.squeeze(0)
        b, _, h, w = x.shape
        x = self.projection(x)

        # CLS tokens
        cls_tokens = repeat(self.cls_tokens, "() n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional embedding
        # new_h = (h+2*self.padding)//self.patch_size
        # new_w = (w+2*self.padding)//self.patch_size
        positions = nn.Parameter(torch.randn(226, self.embed_size))
        x = x + positions

        return x
    