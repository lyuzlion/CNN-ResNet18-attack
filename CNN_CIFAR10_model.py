import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
 
        self.out=nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(DROUPOUT),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(DROUPOUT),
            nn.Linear(256, 10)
        )
 
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        # print(" x shape ",x.size())
        x=x.view(x.size(0), -1)
        output=self.out(x)
        return output

EPOCH = 200
BATCH_SIZE = 128
LR = 0.02
MOMENTUM = 0.9
DROUPOUT = 0.5
DOWNLOAD_CIFAR = True
DELAY_RATE = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
    transforms.RandomCrop(32, padding=4),
])
 
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
])

