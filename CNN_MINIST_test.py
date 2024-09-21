import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import CNN_MNIST_model
from CNN_MNIST_model import EPOCH, BATCH_SIZE, LR, MOMENTUM, DROUPOUT, transform, CNN

test_data=torchvision.datasets.MNIST(
    transform=transform,
    root='./data/mnist',
    train=False,
)


test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn=nn.CrossEntropyLoss()
model = CNN()
model.load_state_dict(torch.load('checkpoint/cnn_mnist.pt', weights_only=False))
model = model.to(device)

model.eval()
correct = 0
test_num = 0
for step, (batch_x, batch_y) in enumerate(test_loader):
    test_output=model(batch_x.to(device))
    # print(test_output)
    loss = loss_fn(test_output, batch_y.to(device))
    y_pred=torch.argmax(test_output,1)
    # print(y_pred)
    correct += sum(y_pred == batch_y.to(device)).item()
    test_num += batch_y.size(0)
print('Accuracy :   ' , correct / test_num)
