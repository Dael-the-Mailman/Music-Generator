import torch
import torch.nn as nn
from critic import Critic
from generator import Generator

import pytorch_lightning as pl

class WGAN_GP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.crit = Critic()
        self.gen = Generator()
    
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    