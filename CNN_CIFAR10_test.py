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
from CNN_CIFAR10_model import EPOCH, BATCH_SIZE, CNN, transform1


test_data=torchvision.datasets.CIFAR10(
    transform=transform1,
    root='./data/cifar10',
    train=False,
)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn=nn.CrossEntropyLoss()
model = CNN()
model.load_state_dict(torch.load('checkpoint/cnn_cifar10.pt', weights_only=False))
model = model.to(device)

model.eval()
valid_loss = 0.0
correct = 0
test_num = 0
for step, (batch_x, batch_y) in enumerate(test_loader):
    test_output=model(batch_x.to(device))
    # print(test_output)
    loss = loss_fn(test_output, batch_y.to(device))
    valid_loss += loss.item() * batch_x.size(0)
    y_pred=torch.argmax(test_output,1)
    # print(y_pred)
    correct += sum(y_pred == batch_y.to(device)).item()
    test_num += batch_y.size(0)
print('Accuracy :   ' , correct / test_num)
