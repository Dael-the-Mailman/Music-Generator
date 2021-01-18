"""
Created a LightningDataModule to load the spectrogram and waveform
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
        waveform, _ = torchaudio.load(song_path)
        specgram = torchaudio.transforms.Spectrogram()(waveform)
        # duration = specgram.shape[2]//221 + 1
        # new_length = duration * 221
        # out = torch.cat((specgram, torch.zeros(2,201,new_length-specgram.shape[2])),2)
        #         # input                                                          , target
        # return [einops.rearrange(out, 'd h (w s) -> d h w s', w=221).unsqueeze(0), waveform]
        return [specgram, waveform]

class ValidDataset(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path,'valid')
        self.songs = os.listdir(self.path)
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, index):
        song_path = os.path.join(self.path, self.songs[index])
        waveform, _ = torchaudio.load(song_path)
        specgram = torchaudio.transforms.Spectrogram()(waveform)
        # duration = specgram.shape[2]//221 + 1
        # new_length = duration * 221
        # out = torch.cat((specgram, torch.zeros(2,201,new_length-specgram.shape[2])),2)
        #         # input                                                          , target
        # return [einops.rearrange(out, 'd h (w s) -> d h w s', w=221).unsqueeze(0), waveform]
        return [specgram, waveform]

class TestDataset(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path,'test')
        self.songs = os.listdir(self.path)
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, index):
        song_path = os.path.join(self.path, self.songs[index])
        waveform, _ = torchaudio.load(song_path)
        specgram = torchaudio.transforms.Spectrogram()(waveform)
        # duration = specgram.shape[2]//221 + 1
        # new_length = duration * 221
        # out = torch.cat((specgram, torch.zeros(2,201,new_length-specgram.shape[2])),2)
        #         # input                                                          , target
        # return [einops.rearrange(out, 'd h (w s) -> d h w s', w=221).unsqueeze(0), waveform]
        return [specgram, waveform]

class LofiDataModule(LightningDataModule):
    def __init__(self, path, batch_size=1):
        super().__init__()
        self.path = path
    
    def train_dataloader(self):
        train_split = TrainDataset(path)
        return DataLoader(train_split, batch_size=self.batch_size)
    
    def val_dataloader(self):
        val_split = ValidDataset(path)
        return DataLoader(val_split, batch_size=self.batch_size)
    
    def test_dataloader(self):
        test_split = TestDataset(path)
        return DataLoader(test_split, batch_size=self.batch_size)