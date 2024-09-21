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
from tqdm import tqdm
from CNN_CIFAR10_model import EPOCH, BATCH_SIZE, LR, MOMENTUM, DROUPOUT, DOWNLOAD_CIFAR, transform, CNN, transform1


train_data = torchvision.datasets.CIFAR10(
    root ='./data/cifar10',
    train = True,
    transform=transform,
    download = DOWNLOAD_CIFAR
)
# print(train_data.data[0].shape)
valid_data=torchvision.datasets.CIFAR10(
    transform=transform1,
    root='./data/cifar10',
    train=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = Data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)

model = CNN().to(device)

def adjust_momentum(optimizer):
    global MOMENTUM
    MOMENTUM = MOMENTUM * 0.8
    for param_group in optimizer.param_groups:
        param_group['momentum'] = MOMENTUM

def adjust_learning_rate(optimizer):
    global LR
    LR = LR * 0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR


optimizer=torch.optim.SGD(model.parameters(), lr=LR, weight_decay=5e-3)
loss_fn=nn.CrossEntropyLoss()
valid_loss_min = np.Inf

for epoch in tqdm(range(1, EPOCH + 1)):
    model.train()
    for step,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output=model(batch_x)
        loss=loss_fn(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    valid_loss = 0.0
    correct = 0
    test_num = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(valid_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            test_output=model(batch_x)

            loss = loss_fn(test_output, batch_y)
            valid_loss += loss.item() * batch_x.size(0)
            y_pred=torch.argmax(test_output,1)
            # print(y_pred)
            correct += (y_pred == batch_y).sum().item()
            test_num += batch_y.size(0)
    print('now epoch :  ', epoch, '     |   accuracy :   ' , correct / test_num)

    if epoch % 10 == 0:
        adjust_momentum(optimizer)
        adjust_learning_rate(optimizer)
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), './checkpoint/cnn_cifar10.pt')
        valid_loss_min = valid_loss