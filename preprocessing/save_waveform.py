import os
import re
import torch
import torchaudio

from tqdm import *

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

IN_PATH = "E:/datasets/youtube/wavfiles"
OUT_PATH = "E:/datasets/youtube/audiotensors"

FIXED_SR = 44100 # Default Sample Rate

# Couldn't make multiprocessed 😭😭😭
songs = os.listdir(IN_PATH)
for song in tqdm(songs):
    song_path = os.path.join(IN_PATH, song)
    waveform, _ = torchaudio.load(song_path)
    new_dir = os.path.join(OUT_PATH, re.sub('.wav', '', song))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for i in range(0, waveform.shape[1]//FIXED_SR):
        torch.save(waveform[:,i*FIXED_SR:(i+1)*FIXED_SR], 
                   os.path.join(new_dir, f"{i}.pt"))