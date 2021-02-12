"""
Implementation of Generator Network for Wasserstein GAN

Based off code from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/model.py
"""
import torch
import torch.nn as nn

class Generator(nn.Sequential):
    def __init__(self, channels_noise, channels, features_g):
        super().__init__(
            # Input N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 3, 1),  # 3
            self._block(features_g * 16, features_g * 8, 3, 3),  # 9
            self._block(features_g * 8, features_g * 4, 7, 7),  # 63
            self._block(features_g * 4, features_g * 2, 7, 7),  # 441
            self._block(features_g * 2, features_g, 10, 10),  # 4410
            # Turns 64x64 -> 128x128
            nn.ConvTranspose1d(
                features_g, channels, kernel_size=10, stride=10, padding=0
            ),
            nn.ReLU()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )