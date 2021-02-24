"""
Implementation of Critic Network for Wasserstein GAN

Based off code from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/model.py
"""
import torch
import torch.nn as nn
import torch.cuda.amp as amp

# class Critic(nn.Sequential):
#     def __init__(self, channels, features_d):
#         super().__init__(
#             # Input: Seconds x channels x 44100
#             nn.Conv1d(channels, features_d, kernel_size=10, stride=10, padding=0),
#             # Blocks below turn 4410 -> 3
#             self._block(features_d, features_d * 2, 10, 10), # 441
#             self._block(features_d * 2, features_d * 4, 7, 7), # 63
#             self._block(features_d * 4, features_d * 8, 7, 7), # 9
#             self._block(features_d * 8, features_d * 16, 3, 3), # 3
#             # Turns 3 into 1
#             nn.Conv1d(features_d*16, 1, kernel_size=3, stride=1, padding=0)
#         )
    
#     def _block(self, in_channels, out_channels, kernel_size, stride, padding=0):
#         return nn.Sequential(
#             nn.Conv1d(
#                 in_channels, out_channels, kernel_size, stride, padding, bias=False,
#             ),
#             nn.InstanceNorm1d(out_channels, affine=True),
#             nn.GELU(),
#         )

class Critic(nn.Module):
    def __init__(self, channels, features_d):
        super().__init__()
        self.crit = nn.Sequential(
            # Input: Seconds x channels x 44100
            nn.Conv1d(channels, features_d, kernel_size=10, stride=10, padding=0),
            # Blocks below turn 4410 -> 3
            self._block(features_d, features_d * 2, 10, 10), # 441
            self._block(features_d * 2, features_d * 4, 7, 7), # 63
            self._block(features_d * 4, features_d * 8, 7, 7), # 9
            self._block(features_d * 8, features_d * 16, 3, 3), # 3
            # Turns 3 into 1
            nn.Conv1d(features_d*16, 1, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        with amp.autocast():
            return self.crit(x)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.GELU(),
        )