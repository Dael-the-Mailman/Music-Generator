import torch
import torch.nn as nn
import sys
sys.path.append('../')
from modules.datamodule import TrainDataset
from modules.patch_embed import *
from modules.pos_enc import *

from einops import rearrange

# train_set = TrainDataset('E:/datasets/youtube/wavfiles')
# print(train_set[0][0].shape)
# print(train_set[0][1].shape)
# trg = train_set[0][1]
# song = train_set[0][0].unsqueeze(0)
# print(song.shape)
# embed = PatchEmbedding()
# embed_song = embed(song)
# print(embed_song.shape)
# pos_enc = PositionalEncoder()(embed_song)
# print(pos_enc.shape)
# layer_size = 4
# dropout = 0.2
# model = nn.Transformer(d_model=360, num_encoder_layers=layer_size, 
#                        num_decoder_layers=layer_size, dropout=dropout, 
#                        activation='gelu', dim_feedforward=512)
# out = model(pos_enc[:-1],pos_enc[1:])
# print(out.shape)
# out2 = rearrange(out, "t n e -> t e n")
# print(out2.shape)
# layer
sample = torch.randn((124,226,360))
print(sample.shape)
x = rearrange(sample,"t n e -> t (n e)")
print(x.shape)
layer = nn.Linear(81360, 44100)
out = layer(sample)
print(out.shape)