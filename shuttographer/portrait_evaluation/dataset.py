from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from utils import *

class PortraitDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label
        
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
    portrait_dataset = ImageFolder(image_folder)

    test_pct = 0.1
    test_size = int(len(portrait_dataset)*test_pct)
    dataset_size = len(portrait_dataset) - test_size

    val_pct = 0.2
    val_size = int(dataset_size*val_pct)
    train_size = dataset_size - val_size

    train_ds, val_ds, test_ds = random_split(portrait_dataset, [train_size, val_size, test_size])

    train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])

    train_dataset = PortraitDataset(train_ds, train_transform)
    val_dataset = PortraitDataset(val_ds, val_transform)
    test_dataset = PortraitDataset(test_ds, test_transform)

    return train_dataset, val_dataset, test_dataset