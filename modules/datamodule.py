import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class LofiDataset(Dataset):
    def __init__(self, )

class LofiDataModule(pl.DataModule):
    def __init__(self, input_path, output_path, batch_size=32):
        self.input = input_path
        self.output = output_path

    def 