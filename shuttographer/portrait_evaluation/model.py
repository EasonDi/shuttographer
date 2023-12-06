import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class QualityModel(nn.Module):
  def __init__(self):
    super(QualityModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 5)
    self.conv2 = nn.Conv2d(32, 64, 5)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv4 = nn.Conv2d(128, 256, 5)
    
    self.fc1 = nn.Linear(256, 9)
    
    self.pool = nn.MaxPool2d(2, 2)
        
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    bs, _, _, _ = x.shape
    x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
    x = F.sigmoid(self.fc1(x))
    return x

class QualityClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(QualityClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the last fully connected layer for classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)