import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm.notebook import tqdm
import torch.optim as optim
from utils import *
from model import *
from dataset import *
import argparse

def train(model, trainloader, optimizer, criterion):
  model.train()
  running_loss = 0.0
  total_error = 0
  counter = 0
  for inputs, labels in trainloader:
    counter += 1
    inputs, labels = inputs.to(device), labels.float().to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))
    running_loss += loss.item()
    total_error += torch.abs(outputs.data.flatten() - labels).sum().item()
    loss.backward()
    optimizer.step()
  epoch_loss = running_loss/counter
  epoch_error = total_error/len(trainloader.dl.dataset)
  return epoch_loss, epoch_error

def validate(model, testloader, criterion):
    model.eval()
    running_loss = 0.0
    total_error = 0
    counter = 0
    for inputs, labels in testloader:
      counter += 1
      inputs, labels = inputs.to(device), labels.float().to(device)
      # forward pass
      outputs = model(inputs)
      # calculate the loss
      loss = criterion(outputs, labels.unsqueeze(1))
      running_loss += loss.item()
      # calculate the accuracy
      total_error += torch.abs(outputs.data.flatten() - labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = running_loss / counter
    epoch_error = total_error/len(testloader.dl.dataset)
    return epoch_loss, epoch_error

if __name__=="__main__":
    batch_size = 64

    # Create DataLoaders
    train_dataset, val_dataset, test_dataset = get_datasets('placeholder')
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)

    # getting default device
    device = get_default_device()

    # moving train dataloader and val dataloader to gpu
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20,
        help='number of epochs to train our network for')
    args = vars(parser.parse_args())
    lr = 1e-3
    epochs = 30
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")

    model = QualityModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_error = train(model, train_dl, optimizer, criterion)
        valid_epoch_loss, valid_epoch_error = validate(model, val_dl, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_error)
        valid_acc.append(valid_epoch_error)
        print(f"Training loss: {train_epoch_loss:.3f}, training error: {train_epoch_error:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation error: {valid_epoch_error:.3f}")
        print('-'*50)

    # save the trained model weights
    save_model(epochs, model, optimizer, criterion)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
