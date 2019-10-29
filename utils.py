import math
import os
import json
import torch
import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(model_name, config):
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def one_hot(y, n_dims):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).cuda()
    return zeros.scatter(scatter_dim, y_tensor, 1)

# dynamic routing
def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    v = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return v

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

