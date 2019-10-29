import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import random

random.seed(2019)

class Attack(object):
    def __init__(self, net, criterion, attack_type, eps):
        self.net = net
        self.criterion = criterion
        self.attack_type = attack_type

        if attack_type not in ["bim", "fgsm"]:
            raise NotImplementedError("Unknown attack type")

        self.eps = eps

    def make(self, x, y, target):
        return getattr(self, self.attack_type)(x, y, target=target)

    def bim(self, x, y, target=None, x_val_min=-1, x_val_max=1):
        out = self.net(x)
        pred = torch.max(out, 1)[1]

        if pred.item() != y.item():
            return None

        eta = torch.zeros_like(x)
        iters = 10
        eps_iter = self.eps / iters
        for i in range(iters):
            nx = x + eta
            nx.requires_grad_()

            out = self.net(nx)

            self.net.zero_grad()
            if target is not None:
                cost = self.criterion(out, target)
            else:
                cost = -self.criterion(out, y)
            cost.backward()

            eta -= eps_iter * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = x + eta
        x_adv.clamp_(x_val_min, x_val_max)

        if target is not None:
            return x_adv.detach(), target

        return x_adv.detach(), y

    def fgsm(self, x, y, target=None, x_val_min=-1, x_val_max=1):
        data = Variable(x.data, requires_grad=True)
        out = self.net(data)
        pred = torch.max(out, 1)[1]

        if pred.item() != y.item():
            return None

        if target is not None:
            cost = self.criterion(out, target)
        else:
            cost = -self.criterion(out, y)

        self.net.zero_grad()
        if data.grad is not None:
            data.grad.data.fill_(0)
        cost.backward()

        data.grad.sign_()
        data = data - self.eps * data.grad
        x_adv = torch.clamp(data, x_val_min, x_val_max)

        if target is not None:
            return x_adv, target

        return x_adv, y

def extract_adv_images(attacker, dataloader, targeted, classes=10):
    adv_images = []
    num_examples = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        curr_x_adv_batch = []
        curr_y_batch = []
        for i in range(len(y)):
            if targeted:
                y_new = y[i] + 1
                if y_new == classes:
                    y_new = 0
                target = y.new_zeros(1)
                target[0] = y_new
                gg = attacker.make(x[i:i+1], y[i:i+1], target=target)
            else:
                gg = attacker.make(x[i:i+1], y[i:i+1], target=None)

            if gg is not None:
                curr_x_adv_batch.append(gg[0])
                curr_y_batch.append(gg[1])
                num_examples += 1

        curr_x_adv_batch = torch.cat(curr_x_adv_batch, dim=0)
        curr_y_batch = torch.cat(curr_y_batch, dim=0)
        adv_images.append((curr_x_adv_batch, curr_y_batch))

        if batch == 20:
            break

    return adv_images, num_examples


