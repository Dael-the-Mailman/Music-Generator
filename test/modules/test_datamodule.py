import os
import sys
import matplotlib.pyplot as plt
import librosa.display as display
sys.path.append('../../')
from modules.datamodule import *
from tqdm import tqdm

train_spec = TrainSpecLoader('E:/datasets/youtube/wavfiles')
valid_spec = ValidSpecLoader('E:/datasets/youtube/wavfiles')
test_spec = TestSpecLoader('E:/datasets/youtube/wavfiles')
print(len(train_spec))
print(len(valid_spec))
print(len(test_spec))
print(train_spec[0][0][0].shape)
print(valid_spec[0][0][0].shape)
print(test_spec[0][0][0].shape)
print(train_spec[0][1][-1].shape)
print(valid_spec[0][1][-1].shape)
print(test_spec[0][1][-1].shape)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(train_spec[0][0][0], ref=np.max)
img = display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=22050,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()
specloader = DataLoader(train_spec, batch_size=1)
num_iterations = len(train_spec)
for i, (src, trg) in tqdm(enumerate(specloader)):
    if (i+1) % 10 == 0:
        print(f"Step: {i+1}/{num_iterations}, \
                inputs: {src[0][0].unsqueeze(0).shape}, \
                outputs: {trg[0][0].unsqueeze(0).shape}")
# train_set = TrainDataset('E:/datasets/youtube/wavfiles')
# valid_set = ValidDataset('E:/datasets/youtube/wavfiles')
# test_set = TestDataset('E:/datasets/youtube/wavfiles')
# dm = LofiDataModule('E:/datasets/youtube/wavfiles')
# print(len(train_set))
# print(len(valid_set))
# print(len(test_set))
# print(train_set[0][0].shape)
# print(valid_set[0][0].shape)
# print(test_set[0][0].shape)
# print(train_set[0][1].shape)
# print(valid_set[0][1].shape)
# print(test_set[0][1].shape)
# train_dl = DataLoader(train_set, batch_size=1)
# num_iterations = len(train_set)
# for i, (src, trg) in tqdm(enumerate(train_dl)):
#     if (i+1) % 10 == 0:
#         print(f"Step: {i+1}/{num_iterations}, \
#                 inputs: {src.squeeze(0).shape}, \
#                 outputs: {trg.squeeze(0).shape}")