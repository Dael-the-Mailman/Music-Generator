import os
import einops
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
from modules.patch_embed import *

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

PATH = 'E:\\datasets\\youtube\\wavfiles\\test'
songs = os.listdir(PATH)
song_path = os.path.join(PATH, songs[0])
print(songs[0])

waveform, _ = torchaudio.load(song_path)
specgram = torchaudio.transforms.Spectrogram()(waveform).log2()[:,:,:]

def fill_spec(spec):
    duration = spec.shape[2]//221 + 1
    new_length = duration * 221
    # out = np.append(spec, np.zeros((2,201,42),dtype=spec.dtype),axis=2)
    out = torch.cat((spec, torch.zeros(2,201,new_length - spec.shape[2])),2)
    return einops.rearrange(out, 'd h (w s) -> d h w s', w=221)

# print(specgram.shape)
# print(specgram.shape[2]//221 + 1)
# print(87*221)
# new_spec = np.append(specgram, np.zeros((2, 201, 42), dtype=np.float32), axis=2)
# new_spec = einops.rearrange(new_spec, 'd h (w s) -> d h w s', w=221)
# print(new_spec.shape)
new_spec = fill_spec(specgram)
new_spec = einops.rearrange(new_spec, 'd h w s -> s d h w').unsqueeze(0)
print(new_spec.shape)
# print(torch.nn.functional.interpolate(new_spec, size=(2,224,224,new_spec.shape[3])).shape)

embed = PatchEmbedding(360)
out = embed(new_spec)

print(out.shape)
# plt.figure()
# plt.imshow(new_spec[0,:,:,0], cmap='gist_heat')
# plt.show()