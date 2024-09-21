import torch

def cw_l2_attack(model, device, x, y, targeted=False, c=1e-4, kappa=0, max_iter=10, learning_rate=0.01) :

    x = x.to(device)     
    y = y.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)
        # print(y.item())
        one_hot_labels = torch.eye(len(outputs[0]))[y.item()].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # print(outputs)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(x, requires_grad=True).to(device)

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1 / 2 * (torch.nn.Tanh()(w) + 1)

        loss1 = torch.nn.MSELoss(reduction='sum')(a, x)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        # print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(torch.nn.Tanh()(w) + 1)

    return attack_images