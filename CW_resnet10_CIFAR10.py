import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.ResNet import ResNet18
from utils.readData import read_dataset
from CW_utils import cw_l2_attack

trainloader, valid_loader, test_loader = read_dataset(batch_size=1,pic_path='./data/cifar10')

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load('./checkpoint/resnet18_cifar10.pt', weights_only=False))
model = model.to(device)
model.eval()

def test(model, device, test_loader):
    correct = 0
    adv_examples = []
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
        # correct += sum(init_pred == target.to(device)).item()
        # test_num += target.size(0)
        # print(init_pred, target)
        if not torch.equal(init_pred, target):
            continue

        perturbed_data = cw_l2_attack(model, device, data, target)
        output = model(perturbed_data)

        final_pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
        # print(final_pred, target)
        if torch.equal(final_pred, target):
            correct += 1
    final_acc = correct / float(len(test_loader))
    print("Test Accuracy = {} / {} = {}".format(correct, len(test_loader), final_acc))

    return final_acc, adv_examples


accuracies = []
examples = []

acc, ex = test(model, device, test_loader)
accuracies.append(acc)
examples.append(ex)
