import torch
import torch.nn as nn
import torch.optim as optim
from wgan_modules.critic import Critic
from wgan_modules.generator import Generator

import pytorch_lightning as pl

class WGAN_GP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, z_dim=100, spec_size=128,
                 channels=2, crit_feat=16, gen_feat=16, n_critic=5, 
                 lambda_gp=10):
        super().__init__()
        self.crit = Critic(channels=channels, features_d=crit_feat)
        self.gen = Generator(channels_noise=z_dim, channels=channels, 
                             features_g=gen_feat)
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
    
    def forward(self, x):
        return self.gen(x)

    def crit_step(self, real, curr_batch_size):
        
        return loss_critic
    
    def gen_step(self, real):
        pass

    def gradient_penalty(critic, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W)
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = critic(interpolated_images)

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
        # Input Shape: (1, duration, 2, 128, 128)
        real, _ = batch.squeeze(0) # (duration, 2, 128, 128)
        opt_c, opt_g = self.optimizers(use_pl_optimizer=True)
        loss_critic = None
        loss_gen = None

        # Generate fake images
        noise = torch.randn(real.shape[0], self.z_dim, 1, 1)
        fake = self.gen(noise)

        if optimizer_idx==0:
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(self.n_critic):
                critic_real = self.critic(real).reshape(-1)
                critic_fake = self.critic(fake).reshape(-1)
                gp = self.gradient_penalty(real, fake)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + 
                    self.lambda_gp * gp
                )
                self.crit.zero_grad()
                self.manual_backward(loss_critic, opt_c, retain_graph=True)
                opt_c.step()
                self.log('Critic Loss', loss_critic)
        
        if optimizer_idx==1:
            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = self.crit(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.gen.zero_grad()
            self.manual_backward(loss_gen, opt_g)
            opt_g.step()
            self.log('Generator Loss', loss_gen)

        return {'Critic Loss': loss_critic, 'Generator Loss': loss_gen}
    
    def configure_optimizers(self):
        lr = self.learning_rate

        opt_g = optim.Adam(self.gen.parameters(), lr=lr, 
                           betas=(0.0, 0.9))
        opt_c = optim.Adam(self.crit.parameters(), lr=lr, 
                           betas=(0.0, 0.9))

        return opt_c, opt_g