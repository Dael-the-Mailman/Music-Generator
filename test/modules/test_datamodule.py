import os
import sys
sys.path.append('../../')
from modules.datamodule import *
from tqdm import tqdm

train_set = TrainDataset('E:/datasets/youtube/wavfiles')
valid_set = ValidDataset('E:/datasets/youtube/wavfiles')
test_set = TestDataset('E:/datasets/youtube/wavfiles')
dm = LofiDataModule('E:/datasets/youtube/wavfiles')
print(len(train_set))
print(len(valid_set))
print(len(test_set))
print(train_set[0][0].shape)
print(valid_set[0][0].shape)
print(test_set[0][0].shape)
print(train_set[0][1].shape)
print(valid_set[0][1].shape)
print(test_set[0][1].shape)
train_dl = DataLoader(train_set, batch_size=1)
num_iterations = len(train_set)
for i, (src, trg) in tqdm(enumerate(train_dl)):
    if (i+1) % 10 == 0:
        print(f"Step: {i+1}/{num_iterations}, \
                inputs: {src.squeeze(0).shape}, \
                outputs: {trg.squeeze(0).shape}")