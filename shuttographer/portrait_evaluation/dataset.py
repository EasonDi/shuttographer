from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
from utils import *
import cv2
import os

class PortraitDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        print('ds = ', ds)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        zeros = np.zeros(9)
        zeros[label] = 1
        label = zeros
        return img, label

class TestDataset(Dataset):
    def __init__(self, test_folder, transform=None):
        self.ds = []
        for label in os.listdir(test_folder):
            for image in os.listdir(test_folder+'/'+label):
                img = cv2.imread(test_folder+'/'+label+'/'+image)
                self.ds.append((image, img, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        zeros = np.zeros(9)
        zeros[label] = 1
        label = zeros
        return image, img, label
        
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
    
def get_datasets(image_folder):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = PortraitDataset(ImageFolder(image_folder+'train'), transform)
    val_dataset = PortraitDataset(ImageFolder(image_folder+'validate'), transform)
    test_dataset = PortraitDataset(ImageFolder(image_folder+'test'), transform)

    return train_dataset, val_dataset, test_dataset