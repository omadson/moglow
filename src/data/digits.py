import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_digits


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DigitsDataset(Dataset):
    def __init__(self):
        self.images = torch.Tensor(load_digits(as_frame=True)['images']).double().to(device)
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min()) * 255
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        imgs = self.images[idx]
        return imgs, 1