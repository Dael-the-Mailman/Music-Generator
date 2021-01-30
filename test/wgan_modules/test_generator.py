"""
Test out the functionality and shape of the generator
"""
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from wgan_modules.generator import Generator

fixed_noise = torch.randn(64, 100, 1, 1)
gen = Generator()
out = gen(fixed_noise)
print(out.shape)