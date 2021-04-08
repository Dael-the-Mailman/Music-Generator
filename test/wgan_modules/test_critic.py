"""
Test out the functionality and shape of the critic network
"""
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from modules.datamodule import TrainSpecLoader
from wgan_modules.critic import Critic

train_dl = DataLoader(TrainSpecLoader('E:/datasets/youtube/wavfiles'), batch_size=1)
critic = Critic(1, 16)

for i, (src, _) in tqdm(enumerate(train_dl)):
    src = torch.from_numpy(np.stack(src))
    print(src.shape)
    out = critic(src)
    print(out.shape)
    break
    
