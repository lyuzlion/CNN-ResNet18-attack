import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from CNN_CIFAR10_model import CNN, BATCH_SIZE, transform1
from FGSM_utils import fgsm_attack

epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

test_data = datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=transform1) # 第一遍写成transform了。。
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load('./checkpoint/cnn_cifar10.pt', weights_only=False))
model.eval()
loss_fn=nn.CrossEntropyLoss().to(device)


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

        loss = loss_fn(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = torch.argmax(output, dim=1)
        if torch.equal(final_pred, target):
            correct += 1

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
plt.xticks(np.arange(0, 0.6, step=0.1))
plt.title("FGSM on CNN (CIFAR10)")
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
#         plt.imshow(ex)
# plt.tight_layout()
# plt.show()