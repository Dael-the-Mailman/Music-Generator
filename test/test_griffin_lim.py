import os
import librosa
import librosa.display as display
import soundfile
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import einops

DIR = 'E:/datasets/youtube/wavfiles/train/Caleb - 504.wav'

y, sr = librosa.load(DIR, duration=60.0)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)

print(S.shape)

y_prime = librosa.feature.inverse.mel_to_audio(S, sr=sr)
soundfile.write('E:/datasets/youtube/Mel Caleb - 504.wav', y_prime, sr)

# fig, ax = plt.subplots()
# fig.set_size_inches(16,9)
# S_dB = librosa.power_to_db(S, ref=np.max)
# img = display.specshow(S_dB, x_axis='time',
#                          y_axis='mel', sr=sr,
#                          fmax=8000, ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# plt.show()