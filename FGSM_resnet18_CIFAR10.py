import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.readData import read_dataset
from utils.ResNet import ResNet18
from FGSM_utils import fgsm_attack

epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
train_loader, valid_loader, test_loader = read_dataset(batch_size=1,pic_path='./data/cifar10')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load('./checkpoint/resnet18_cifar10.pt', weights_only=False))
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss().to(device)

def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        data.requires_grad = True

        output = model(data)
        init_pred = torch.argmax(output, dim=1)  # get the index of the max log-probability

        if not torch.equal(init_pred, target):
            continue

        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = torch.argmax(output, dim=1)

        if torch.equal(final_pred, target):
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples


accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.5, step=0.1))
plt.title("FGSM on ResNet18 (CIFAR10)")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# 在每个epsilon上绘制几个对抗样本的例子
# cnt = 0
# plt.figure(figsize=(8, 10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons), len(examples[0]), cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig, adv, ex = examples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
        
#         plt.imshow(np.transpose(ex, (2, 1, 0)))
# plt.tight_layout()
# plt.show()