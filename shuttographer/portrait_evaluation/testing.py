import torch
import gdown
import os
import torchvision.transforms as transforms
import cv2
import pandas
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
from utils import *
from model import *
from dataset import *

device = ('cpu')
portrait_model = QualityClassifier().to(device)
model_url = 'https://drive.google.com/uc?id=1oQKuKN5N33KfSMptlNuNHWqmkrub1SAs'
out = gdown.download(model_url, os.path.join('.', 'portrait_evaluation_model.pth'))
if out is None:
    print('failed to download model from drive')
portrait_model.load_state_dict(torch.load('portrait_evaluation_model.pth')['model_state_dict'])
portrait_model.eval()
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_quality_score(image):
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        output = portrait_model(image.to(device))
    return torch.max(output, 1)[1].item()

if __name__=='__main__':
    names = []
    pred_scores = []
    true_scores = []
    total_correct = 0
    within_1 = 0
    batch_size = 64
    # CHANGE PATH to test dataset folder
    test_dataset = TestDataset('/home/bk632/portrait_data/test', transform)
    for i in range(len(test_dataset)):
        image_name, img, label = test_dataset[i]
        outputs = portrait_model(img.unsqueeze(0))
        preds, correct = torch.max(outputs, 1)[1], torch.max(torch.from_numpy(label), 0)[1]
        total_correct += (preds==correct).sum().item()
        if abs(preds.item()-correct.item()) <= 1:
            within_1 += 1
        names.append(image_name)
        pred_scores.append(preds.item())
        true_scores.append(correct.item())
    print("accuracy = ", total_correct/len(test_dataset))
    print("accuracy with tolerance of +/- 1 = ", within_1/len(test_dataset))
    data = pandas.DataFrame({'name':names, 'predicted_score':pred_scores, 'true_score':true_scores})
    data.to_csv('test_results.csv')
