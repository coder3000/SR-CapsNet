import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *

from utils import weights_init


class SmallNet(nn.Module):
    def __init__(self, cfg_data, mode='SR'):
        super(SmallNet, self).__init__()
        channels, classes = cfg_data['channels'], cfg_data['classes']
        self.conv1 = nn.Conv2d(channels, 256, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.mode = mode

        self.num_caps = 16

        planes = 16
        last_size = 6
        if self.mode == 'SR':
            self.conv_a = nn.Conv2d(256, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(256, self.num_caps*planes, kernel_size=5, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(self.num_caps)
            self.bn_pose = nn.BatchNorm2d(self.num_caps*planes)

            self.conv_caps = SelfRouting2d(self.num_caps, self.num_caps, planes, planes, kernel_size=3, stride=2, padding=1, pose_out=True)
            self.bn_pose_conv_caps = nn.BatchNorm2d(self.num_caps*planes)

            self.fc_caps = SelfRouting2d(self.num_caps, classes, planes, 1, kernel_size=last_size, padding=0, pose_out=False)

        elif self.mode == 'DR':
            self.conv_pose = nn.Conv2d(256, self.num_caps*planes, kernel_size=5, stride=1, padding=1, bias=False)
            # self.bn_pose = nn.BatchNorm2d(self.num_caps*planes)

            self.conv_caps = DynamicRouting2d(self.num_caps, self.num_caps, 16, 16, kernel_size=3, stride=2, padding=1)
            nn.init.normal_(self.conv_caps.W, 0, 0.5)

            self.fc_caps = DynamicRouting2d(self.num_caps, classes, 16, 16, kernel_size=last_size, padding=0)
            nn.init.normal_(self.fc_caps.W, 0, 0.05)

        elif self.mode == 'EM':
            self.conv_a = nn.Conv2d(256, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(256, self.num_caps*16, kernel_size=5, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(self.num_caps)
            self.bn_pose = nn.BatchNorm2d(self.num_caps*16)

            self.conv_caps = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
            self.bn_pose_conv_caps = nn.BatchNorm2d(self.num_caps*planes)

            self.fc_caps = EmRouting2d(self.num_caps, classes, 16, kernel_size=last_size, padding=0)

        else:
            raise NotImplementedError

        self.apply(weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        if self.mode == 'DR':
            # pose = self.bn_pose(self.conv_pose(out))
            pose = self.conv_pose(out)

            b, c, h, w = pose.shape
            pose = pose.permute(0, 2, 3, 1).contiguous()
            pose = squash(pose.view(b, h, w, self.num_caps, 16))
            pose = pose.view(b, h, w, -1)
            pose = pose.permute(0, 3, 1, 2)

            pose = self.conv_caps(pose)

            out = self.fc_caps(pose)
            out = out.view(b, -1, 16)
            out = out.norm(dim=-1)

        elif self.mode == 'EM':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            a, pose = self.conv_caps(a, pose)
            pose = self.bn_pose_conv_caps(pose)

            a, _ = self.fc_caps(a, pose)
            out = a.view(a.size(0), -1)

        elif self.mode == 'SR':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            a, pose = self.conv_caps(a, pose)
            pose = self.bn_pose_conv_caps(pose)

            a, _ = self.fc_caps(a, pose)

            out = a.view(a.size(0), -1)
            out = out.log()

        return out


