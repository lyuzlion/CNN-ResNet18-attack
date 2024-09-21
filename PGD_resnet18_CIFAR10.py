import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.ResNet import ResNet18
from utils.readData import read_dataset
from PGD_utils import PGD_attack

iterations = [0, 3, 6, 9, 12]

train_loader, valid_loader, test_loader = read_dataset(batch_size=1,pic_path='./data/cifar10')

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load('./checkpoint/resnet18_cifar10.pt', weights_only=False))
model = model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss().to(device)

def test(model, device, test_loader, iterations):
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


        perturbed_data = PGD_attack(model, device, data, target, criterion, iterations)

        output = model(perturbed_data)

        final_pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
        # print(final_pred, target)
        if torch.equal(final_pred, target):
            correct += 1
            if (iterations == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct / float(len(test_loader))
    print("Iterations: {}\tTest Accuracy = {} / {} = {}".format(iterations, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


accuracies = []
examples = []

for iter in iterations:
    acc, ex = test(model, device, test_loader, iter)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(iterations, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.6, step=0.1))
plt.title("PGD on ResNet18 (CIFAR10)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
