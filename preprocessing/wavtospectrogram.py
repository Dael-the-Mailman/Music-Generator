import os
import torch
import torchaudio

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

PATH = "E:/datasets/youtube/wavfiles"

songs = os.listdir(PATH)
print(songs)
data, sample_rate = torchaudio.load(os.path.join(PATH, songs[0]))
print(data.size())