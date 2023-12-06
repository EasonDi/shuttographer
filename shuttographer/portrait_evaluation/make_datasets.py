from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
import os
import shutil

if __name__=="__main__":
    data_folder = '/home/bk632/indoor_portraits_labelled'
    new_folder = '/home/bk632/portrait_data/'
    images = []
    labels = {}
    for label in os.listdir(data_folder):
        for image in os.listdir(data_folder+'/'+label):
            images.append(image)
            labels[image] = label
    train_size, val_size, test_size = (1397-(140+280)), 280, 140
    train_set, val_set, test_set = random_split(images, [train_size, val_size, test_size])
    for image in train_set:
        shutil.copy(data_folder+'/'+labels[image]+'/'+image, new_folder+'train/'+labels[image]+'/'+image)
    for image in val_set:
        shutil.copy(data_folder+'/'+labels[image]+'/'+image, new_folder+'validate/'+labels[image]+'/'+image)
    for image in test_set:
        shutil.copy(data_folder+'/'+labels[image]+'/'+image, new_folder+'test/'+labels[image]+'/'+image)
