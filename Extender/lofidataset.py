import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class LofiDataset(Dataset):
    def __init__(self, path, batch_size, n_mels=128):
        self.path = path
        self.n_mels = n_mels
        self.batch_size = batch_size

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        sample, sample_rate = torchaudio.load(
            os.path.join(self.path, os.listdir(self.path)[idx]))
        melspec = MelSpectrogram(sample_rate, n_mels=self.n_mels)(sample)
        split = list(torch.split(melspec, self.n_mels, dim=2))
        # Concatenate zeros to the last tensor if need be
        dead_space = self.n_mels - split[-1].shape[2]
        if dead_space > 0:
            space_tensor = torch.zeros(
                (split[0].shape[0], self.n_mels, dead_space))
            split[-1] = torch.cat((split[-1], space_tensor), dim=2)
        stack = torch.stack(split)
        return list(torch.split(stack, self.batch_size, dim=0))
