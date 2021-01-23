import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

import sys
sys.path.append('../../')
from modules.datamodule import TrainDataset
from modules.patch_embed import PatchEmbedding
from modules.pos_enc import PositionalEncoder
from modules.attention import MultiHeadAttention

train_set = TrainDataset('E:/datasets/youtube/wavfiles')
song = train_set[0][0].unsqueeze(0)
print(song.shape)
embed = PatchEmbedding()
embed_song = embed(song)
print(embed_song.shape)
pos_enc = PositionalEncoder()(embed_song)
print(pos_enc.shape)
q = nn.Linear(360, 360)(pos_enc)
q = rearrange(q, "b n (h d) -> b h n d", h=8)
k = nn.Linear(360, 360)(pos_enc)
k = rearrange(k, "b n (h d) -> b h n d", h=8)
print(q.shape)
print(k.shape)
energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
print(energy.shape)
attention = MultiHeadAttention()(pos_enc)
print(attention.shape)
print("Success 😎")