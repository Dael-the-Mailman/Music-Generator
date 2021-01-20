import sys
sys.path.append('../../')
from modules.datamodule import TrainDataset
from modules.patch_embed import *
from modules.pos_enc import *

train_set = TrainDataset('E:/datasets/youtube/wavfiles')
song = train_set[0][0].unsqueeze(0)
print(song.shape)
embed = PatchEmbedding()
embed_song = embed(song)
print(embed_song)
print(embed_song.shape)
pos_enc = PositionalEncoder()(embed_song)
print(pos_enc)
print(pos_enc.shape)
print("Success 😎")