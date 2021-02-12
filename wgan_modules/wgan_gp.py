"""
Implemented Wasserstein GAN in PytorchLightning

Based on: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from wgan_modules.critic import Critic
from wgan_modules.generator import Generator

import pytorch_lightning as pl

class WGAN_GP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, z_dim=100, channels=2, 
                 crit_feat=16, gen_feat=16, n_critic=5, lambda_gp=10):
        super().__init__()
        self.crit = Critic(channels=channels, features_d=crit_feat)
        self.gen = Generator(channels_noise=z_dim, channels=channels, 
                             features_g=gen_feat)
        self.z_dim = z_dim
        self.lr = learning_rate
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
    
    def forward(self, x):
        return self.gen(x)

    def crit_loss(self, x):
        pass

    def gen_loss(self, x):
        pass

    def crit_step(self, x):
        pass

    def gen_step(self, x):
        pass
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def configure_optimizers(self):
        opt_g = optim.Adam(self.gen.parameters(), lr=self.lr)
        opt_c = optim.Adam(self.crit.parameters(), lr=self.lr)
        
        return [opt_g, opt_c], []