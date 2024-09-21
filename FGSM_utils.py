import torch

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image