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
                 crit_feat=16, gen_feat=16, n_critic=5, lambda_gp=10,):
        super().__init__()
        self.crit = Critic(channels=channels, features_d=crit_feat).to("cuda")
        self.gen = Generator(channels_noise=z_dim, channels=channels, 
                             features_g=gen_feat).to("cuda")
        self.z_dim = z_dim
        self.lr = learning_rate
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.fixed_noise = torch.randn(32, z_dim, 1)
    
    def forward(self, x):
        return self.gen(x)

    def gradient_penalty(self, real, fake):
        BATCH_SIZE, C, S = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, S)
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = self.crit(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_g, opt_c = self.optimizers()
        real, _ = batch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        real = real.squeeze(0).to(device)
        
        result = None
        if optimizer_idx == 0:
            curr_batch_size = real.shape[0]
            noise = torch.randn(curr_batch_size, self.z_dim, 1).to(device)
            fake = self.gen(noise)
            gen_fake = self.crit(fake).reshape(-1)
            result = -torch.mean(gen_fake)
            self.gen.zero_grad()
            self.manual_backward(result, opt_g)
            opt_g.step()

        if optimizer_idx == 1:
            for _ in range(self.n_critic):
                curr_batch_size = real.shape[0]
                noise = torch.randn(curr_batch_size, self.z_dim, 1).to(device)
                fake = self.gen(noise)
                crit_real = self.crit(real).reshape(-1)
                crit_fake = self.crit(fake).reshape(-1)
                gp = self.gradient_penalty(real, fake)
                result = (
                    -(torch.mean(crit_real) - torch.mean(crit_fake)) + self.lambda_gp * gp
                )
                self.crit.zero_grad()
                self.manual_backward(result, opt_c, retain_graph=True)
                opt_c.step()

        return result

    def configure_optimizers(self):
        opt_g = optim.Adam(self.gen.parameters(), lr=self.lr)
        opt_c = optim.Adam(self.crit.parameters(), lr=self.lr)
        
        return [opt_g, opt_c], []