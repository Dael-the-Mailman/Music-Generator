"""
Implementation of Generator Network for Wasserstein GAN

Based off code from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/model.py
"""
import torch
import torch.nn as nn

class Generator(nn.Sequential):
    def __init__(self, channels_noise=100, channels=2, features_g=16):
        super().__init__(
            # Input N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            self._block(features_g * 2, features_g, 4, 2, 1),  # 64x64
            # Turns 64x64 -> 128x128
            nn.ConvTranspose2d(
                features_g, channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )