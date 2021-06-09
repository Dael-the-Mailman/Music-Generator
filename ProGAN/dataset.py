import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

class LofiDataset(Dataset):
    def __init__(self, path, n_mels=128):
        self.path = path
        self.n_mels = n_mels

    def __len__(self):
        return len(os.listdir(self.path))
        
    def __getitem__(self, idx):
        sample, sample_rate = torchaudio.load(os.path.join(self.path, os.listdir(self.path)[idx]))
        melspec = MelSpectrogram(sample_rate, n_mels=self.n_mels)(sample)
        split = torch.split(melspec, self.n_mels,dim=2)
        stack = torch.stack(split[:-1]) # Last bit almost always different shape
        return stack