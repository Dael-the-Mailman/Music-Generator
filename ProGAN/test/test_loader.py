import os
import torch
import torchaudio
from torch.utils.data import Dataset, dataset
from torchaudio.transforms import MelSpectrogram
# import torchvision.transforms as transforms

PATH = 'D:\\datasets\\youtube\\all'
NUM_WORKERS = 4
class LofiDataset(Dataset):
    def __init__(self, path, n_mels=128):
        self.path = path
        self.n_mels = n_mels
        # self.transform = transforms.Compose([

        # ])

    def __len__(self):
        return len(os.listdir(PATH))
        
    def __getitem__(self, idx):
        sample, sample_rate = torchaudio.load(os.path.join(self.path, os.listdir(PATH)[idx]))
        melspec = MelSpectrogram(sample_rate, n_mels=self.n_mels)(sample)
        split = torch.split(melspec, self.n_mels,dim=2)
        stack = torch.stack(split[:-1]) # Last bit almost always different shape
        return stack

dataset = LofiDataset(PATH)

data = dataset[0]
print(data.shape)
print(torch.max(data))

# print(len(dataset))
# print(dataset[0])