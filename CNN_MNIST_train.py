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
from CNN_MNIST_model import EPOCH, BATCH_SIZE, LR, MOMENTUM, DROUPOUT, DOWNLOAD_MNIST, transform, CNN

train_data = torchvision.datasets.MNIST(
    root ='./data/mnist',
    train = True,
    transform=transform,
    download = DOWNLOAD_MNIST
)

valid_data = torchvision.datasets.MNIST(
    root ='./data/mnist',
    train = False,
    transform = transform,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = Data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)

model = CNN().to(device)

optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
loss_fn=nn.CrossEntropyLoss()

valid_loss_min = np.Inf

for epoch in tqdm(range(1, EPOCH + 1)):
    model.train()
    for step,(batch_x, batch_y) in enumerate(train_loader):
 
        output=model(batch_x.to(device))
        # print(output.size(0), batch_y.size(0))

        loss=loss_fn(output,batch_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    model.eval()
    valid_loss = 0.0
    correct = 0
    test_num = 0
    for step, (batch_x, batch_y) in enumerate(valid_loader):
        test_output=model(batch_x.to(device))
        # print(test_output)
        loss = loss_fn(test_output, batch_y.to(device))
        valid_loss += loss.item() * batch_x.size(0)
        y_pred=torch.argmax(test_output,1)
        # print(y_pred)
        correct += sum(y_pred == batch_y.to(device)).item()
        test_num += batch_y.size(0)
    print('now epoch :  ', epoch, '     |   accuracy :   ' , correct / test_num)

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/cnn_mnist.pt')
        valid_loss_min = valid_loss
 
