import os
import librosa
import librosa.display as display
import soundfile
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import einops

from tqdm import tqdm

DIR = 'E:/datasets/youtube/wavfiles/train/Caleb - 504.wav'

y, sr = librosa.load(DIR, duration=60.0)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)

def chunkify(arr, chunk_size):
    for i in range(0, arr.shape[1], chunk_size):
        yield arr[:, i:i+chunk_size]

print(S.shape)

S_prime = list(chunkify(S, 224)) # np.array_split(S, S.shape[1]//128, axis=1)

print(S_prime[-1].shape)

print("Inversing spectrogram")
try:
    y_prime = []
    for chunk in tqdm(S_prime):
        out = librosa.feature.inverse.mel_to_audio(chunk, sr=sr)
        y_prime.append(out)
        print(out.shape)
except Exception as e:
    print("Inversion failed 😭")
    print(e)

print("Writing to file")
try:
    soundfile.write('E:/datasets/youtube/Distributed Mel Caleb - 504.wav', np.concatenate(y_prime), sr)
except Exception as e:
    print("Write failed 😭")
    print(e)
# y_prime = librosa.feature.inverse.mel_to_audio(S, sr=sr)
# soundfile.write('E:/datasets/youtube/Mel Caleb - 504.wav', y_prime, sr)

# fig, ax = plt.subplots()
# fig.set_size_inches(16,9)
# S_dB = librosa.power_to_db(S, ref=np.max)
# img = display.specshow(S_dB, x_axis='time',
#                          y_axis='mel', sr=sr,
#                          fmax=8000, ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# plt.show()