import torch.nn as nn
import numpy as np
import torch


def pgd_attack(model, optimizer,images, labels, eps=0.3, alpha=2/255, iters=40) :
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        optimizer.zero_grad()
        cost = loss(outputs, labels).cuda()
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images