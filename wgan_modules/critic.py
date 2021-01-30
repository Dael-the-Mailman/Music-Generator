"""
Implementation of Critic Network for Wasserstein GAN

Based off code from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/model.py
"""
import torch
import torch.nn as nn

class Critic(nn.Sequential):
    def __init__(self, channels=2, features_d=16):
        super().__init__(
            # Input: Seconds x channels x 128 x 128
            nn.Conv2d(channels, features_d, kernel_size=4, stride=2, padding=1),
            # Blocks below turn 128x128 -> 4x4
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            # Turns 4x4 to 1x1
            nn.Conv2d(features_d*16, 1, kernel_size=4, stride=2, padding=0)
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )