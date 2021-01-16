import os
import sys
sys.path.append('../../')
from modules.datamodule import *

train_set = TrainDataset('E:/datasets/youtube/wavfiles')
valid_set = ValidDataset('E:/datasets/youtube/wavfiles')
test_set = TestDataset('E:/datasets/youtube/wavfiles')
dm = LofiDataModule('E:/datasets/youtube/wavfiles')
print(len(train_set))
print(len(valid_set))
print(len(test_set))