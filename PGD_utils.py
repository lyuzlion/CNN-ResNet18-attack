import torch

def PGD_attack(model, device, x, y, loss_fn, iterations, epsilon=0.15, alpha=0.1):
    """
    inputs:
        model: model to be attacked
        x: input image
        y: label
        iterations: number of iterations for PGD attack
        epsilon: clipping threshold for PGD
        alpha: step size for PGD attack

    outputs:
        x': the perturbed image
        delta: the added perturbation
        y': prediction of the model for the perturbed input
    """
    x.requires_grad = True
    init_x = x.clone().detach()
    for i in range(iterations):
        x.requires_grad = True
        predictions = model(x)
        loss = loss_fn(predictions, y)
        model.zero_grad()
        loss.backward()
        delta = alpha * x.grad.detach().sign()
        x = x.detach() + delta.detach()
        x = torch.min(torch.max(x, init_x - epsilon), init_x + epsilon)
        #   x = torch.clamp(x, 0, 1)
    return x