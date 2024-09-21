import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from CNN_CIFAR10_model import CNN, transform1
from PGD_utils import PGD_attack

iterations = [0, 3, 6, 9, 12]

test_data = datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=transform1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load('./checkpoint/cnn_cifar10.pt', weights_only=False))
loss_fn=nn.CrossEntropyLoss().to(device)
model.eval()

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


        perturbed_data = PGD_attack(model, device, data, target, loss_fn, iterations)

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
plt.title("PGD on CNN (CIFAR10)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
