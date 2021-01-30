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

        # spectrogram calculations
        specgram = torchaudio.transforms.MelSpectrogram()(waveform)
        duration = specgram.shape[2]//128 + 1
        new_length = duration * 128
        spec_out = torch.cat((specgram, torch.zeros(2,128,new_length-specgram.shape[2])),2)
        return [einops.rearrange(spec_out, 'd h (w s) -> s d h w', w=128), 
                einops.rearrange(wave_out, 'd (w s) -> s d w', w=44100)]
        # return [specgram, waveform]

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

        # spectrogram calculations
        specgram = torchaudio.transforms.Spectrogram()(waveform)
        # duration = specgram.shape[2]//221 + 1
        new_length = duration * 221
        spec_out = torch.cat((specgram, torch.zeros(2,201,new_length-specgram.shape[2])),2)
        return [einops.rearrange(spec_out, 'd h (w s) -> s d h w', w=221), 
                einops.rearrange(wave_out, 'd (w s) -> s d w', w=44100)]
        # return [specgram, waveform]

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

        # spectrogram calculations
        specgram = torchaudio.transforms.Spectrogram()(waveform)
        # duration = specgram.shape[2]//221 + 1
        new_length = duration * 221
        spec_out = torch.cat((specgram, torch.zeros(2,201,new_length-specgram.shape[2])),2)
        return [einops.rearrange(spec_out, 'd h (w s) -> s d h w', w=221), 
                einops.rearrange(wave_out, 'd (w s) -> s d w', w=44100)]
        # return [specgram, waveform]

class LofiDataModule(LightningDataModule):
    def __init__(self, path):
        super().__init__()
        self.path = path
    
    def train_dataloader(self):
        train_split = TrainDataset(path)
        return DataLoader(train_split)
    
    def val_dataloader(self):
        val_split = ValidDataset(path)
        return DataLoader(val_split)
    
    def test_dataloader(self):
        test_split = TestDataset(path)
        return DataLoader(test_split)