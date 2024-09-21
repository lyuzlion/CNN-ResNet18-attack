import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
 
        self.out=nn.Sequential(
            nn.Dropout(DROUPOUT),
            nn.Linear(64 * 7 * 7, 200),
            nn.ReLU(),
            nn.Dropout(DROUPOUT),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(DROUPOUT),
            nn.Linear(200, 10)
        )
 
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0), -1)
        output=self.out(x)
        return output

EPOCH = 75
BATCH_SIZE = 128
LR = 0.004
MOMENTUM = 0.9
DROUPOUT = 0.5
DOWNLOAD_MNIST = True


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
])
 



