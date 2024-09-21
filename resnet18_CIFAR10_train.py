import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size,pic_path='./data/cifar10')

model = ResNet18()

model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

epochs = 250
valid_loss_min = np.Inf
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, epochs + 1)):

    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    if counter / 10 == 1:
        counter = 0
        lr = lr * 0.5

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data).to(device)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)    
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    
    print('now epoch :  ', epoch, '     |   accuracy :   ' , right_sample / total_sample)
    accuracy.append(right_sample/total_sample)
 
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
        
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1

