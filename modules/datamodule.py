"""
Created a LightningDataModule to load the spectrogram and waveform

:param path: path to folder containing train, valid, and test folders
"""
import os
import einops
import torch
import torchaudio
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

class TrainDataset(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path,'train')
        self.songs = os.listdir(self.path)
    
    def __len__(self):
        return len(self.songs)

    def __getitem__(self, index):
        song_path = os.path.join(self.path, self.songs[index])
        # waveform calculations
        waveform, _ = torchaudio.load(song_path)
        duration = waveform.shape[1]//44100 + 1
        new_length = 44100*duration
        wave_out = torch.cat((waveform, torch.zeros(2,new_length-waveform.shape[1])),1)
        wave_out = einops.rearrange(wave_out, "c (t s) -> t c s", s=44100)

        return [wave_out[:-1], wave_out[1:]]

class ValidDataset(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path,'valid')
        self.songs = os.listdir(self.path)
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, index):
        song_path = os.path.join(self.path, self.songs[index])
        # waveform calculations
        waveform, _ = torchaudio.load(song_path)
        duration = waveform.shape[1]//44100 + 1
        new_length = 44100*duration
        wave_out = torch.cat((waveform, torch.zeros(2,new_length-waveform.shape[1])),1)
        wave_out = einops.rearrange(wave_out, "c (t s) -> t c s", s=44100)

        return [wave_out[:-1], wave_out[1:]]

class TestDataset(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path,'test')
        self.songs = os.listdir(self.path)
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, index):
        song_path = os.path.join(self.path, self.songs[index])
        # waveform calculations
        waveform, _ = torchaudio.load(song_path)
        duration = waveform.shape[1]//44100 + 1
        new_length = 44100*duration
        wave_out = torch.cat((waveform, torch.zeros(2,new_length-waveform.shape[1])),1)
        wave_out = einops.rearrange(wave_out, "c (t s) -> t c s", s=44100)

        return [wave_out[:-1], wave_out[1:]]

class LofiDataModule(LightningDataModule):
    def __init__(self, path):
        super().__init__()
        self.path = path
    
    def train_dataloader(self):
        train_split = TrainDataset(self.path)
        return DataLoader(train_split)
    
    def val_dataloader(self):
        val_split = ValidDataset(self.path)
        return DataLoader(val_split)
    
    def test_dataloader(self):
        test_split = TestDataset(self.path)
        return DataLoader(test_split)