import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

from utils import one_hot


class DynamicRoutingLoss(nn.Module):
    def __init(self):
        super(DynamicRoutingLoss, self).__init()

    def forward(self, x, target):
        target = one_hot(target, x.shape[1])

        left = F.relu(0.9 - x) ** 2
        right = F.relu(x - 0.1) ** 2

        margin_loss = target * left + 0.5 * (1. - target) * right
        margin_loss = margin_loss.sum(dim=1).mean()
        return margin_loss


class EmRoutingLoss(nn.Module):
    def __init__(self, max_epoch):
        super(EmRoutingLoss, self).__init__()
        self.margin_init = 0.2
        self.margin_step = 0.2 / max_epoch
        self.max_epoch = max_epoch

    def forward(self, x, target, epoch=None):
        if epoch is None:
            margin = 0.9
        else:
            margin = self.margin_init + self.margin_step * min(epoch, self.max_epoch)

        b, E = x.shape
        at = x.new_zeros(b)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum(dim=1).mean()
        return loss
