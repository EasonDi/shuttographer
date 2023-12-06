from model import QualityClassifier
import torch
import gdown
import os
import torchvision.transforms as transforms
import cv2

device = ('cuda' if torch.cuda.is_available() else 'cpu')
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
    image = cv2.imread('/home/bk632/portrait_data/test/0/32_Indoor_Scene_0.jpg')
    print(get_quality_score(image))
