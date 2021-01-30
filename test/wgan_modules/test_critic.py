"""
Test out the functionality and shape of the critic network
"""
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from modules.datamodule import TrainDataset
from wgan_modules.critic import Critic

train_dl = DataLoader(TrainDataset('E:/datasets/youtube/wavfiles'), batch_size=1)
critic = Critic()

for i, (src, _) in tqdm(enumerate(train_dl)):
    out = critic(src.squeeze(0))
    print(out.shape)
    break
    
