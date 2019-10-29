import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules import *
from utils import weights_init


class ConvNet(nn.Module):
    def __init__(self, planes, cfg_data, num_caps, caps_size, depth, mode):
        caps_size = 16
        super(ConvNet, self).__init__()
        channels, classes = cfg_data['channels'], cfg_data['classes']
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth
        self.mode = mode

        self.layers = nn.Sequential(
            nn.Conv2d(channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*8),
            nn.ReLU(True),
        )

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        #========= ConvCaps Layers
        for d in range(1, depth):
            if self.mode == 'DR':
                self.conv_layers.append(DynamicRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1))
                nn.init.normal_(self.conv_layers[0].W, 0, 0.5)
            elif self.mode == 'EM':
                self.conv_layers.append(EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, stride=1, padding=1))
                self.norm_layers.append(nn.BatchNorm2d(4*4*num_caps))
            elif self.mode == 'SR':
                self.conv_layers.append(SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1, pose_out=True))
                self.norm_layers.append(nn.BatchNorm2d(planes*num_caps))
            else:
                break

        final_shape = 4

        # DR
        if self.mode == 'DR':
            self.conv_pose = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
            self.fc = DynamicRouting2d(num_caps, classes, caps_size, caps_size, kernel_size=final_shape, padding=0)
            # initialize so that output logits are in reasonable range (0.1-0.9)
            nn.init.normal_(self.fc.W, 0, 0.1)

        # EM
        elif self.mode == 'EM':
            self.conv_a = nn.Conv2d(8*planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(num_caps)
            self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
            self.fc = EmRouting2d(num_caps, classes, caps_size, kernel_size=final_shape, padding=0)

        # SR
        elif self.mode == 'SR':
            self.conv_a = nn.Conv2d(8*planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(num_caps)
            self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
            self.fc = SelfRouting2d(num_caps, classes, caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

        # avg pooling
        elif self.mode == 'AVG':
            self.pool = nn.AvgPool2d(final_shape)
            self.fc = nn.Linear(8*planes, classes)

        # max pooling
        elif self.mode == 'MAX':
            self.pool = nn.MaxPool2d(final_shape)
            self.fc = nn.Linear(8*planes, classes)

        elif self.mode == 'FC':
            self.conv_ = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_ = nn.BatchNorm2d(num_caps*caps_size)

            self.fc = nn.Linear(num_caps*caps_size*final_shape*final_shape, classes)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)

        # DR
        if self.mode == 'DR':
            pose = self.bn_pose(self.conv_pose(out))

            b, c, h, w = pose.shape
            pose = pose.permute(0, 2, 3, 1).contiguous()
            pose = squash(pose.view(b, h, w, self.num_caps, self.caps_size))
            pose = pose.view(b, h, w, -1)
            pose = pose.permute(0, 3, 1, 2)

            for m in self.conv_layers:
                pose = m(pose)

            out = self.fc(pose)
            out = out.view(b, -1, self.caps_size)
            out = out.norm(dim=-1)

        # EM
        elif self.mode == 'EM':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)

            a, _ = self.fc(a, pose)
            out = a.view(a.size(0), -1)

        # ours
        elif self.mode == 'SR':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)

            a, _ = self.fc(a, pose)
            out = a.view(a.size(0), -1)
            out = out.log()

        elif self.mode == 'AVG' or self.mode =='MAX':
            out = self.pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        elif self.mode == 'FC':
            out = F.relu(self.bn_(self.conv_(out)))
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out

