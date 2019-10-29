import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import squash


eps = 1e-12


class DynamicRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=1, stride=1, padding=1, iters=3):
        super(DynamicRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters
        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, pose):
        # x: [b, AC, h, w]
        b, _, h, w = pose.shape
        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        # [b, l, kkA, BD]
        pose_out = torch.matmul(self.W, pose).squeeze(-1)
        # [b, l, kkA, B, D]
        pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B, 1]
        b = pose.new_zeros(b, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            c = torch.softmax(b, dim=3)

            # [b, l, 1, B, D]
            s = (c * pose_out).sum(dim=2, keepdim=True)
            # [b, l, 1, B, D]
            v = squash(s)

            b = b + (v * pose_out).sum(dim=-1, keepdim=True)

        # [b, l, B, D]
        v = v.squeeze(2)
        # [b, l, BD]
        v = v.view(v.shape[0], l, -1)
        # [b, BD, l]
        v = v.transpose(1,2).contiguous()

        oh = ow = math.floor(l**(1/2))

        # [b, BD, oh, ow]
        return v.view(v.shape[0], -1, oh, ow)


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2*math.pi)

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_*(self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq*self.ln_2pi), dim=-1) \
                    - torch.sum((v - mu)**2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i+1)) 
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out


class SelfRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(SelfRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.pose_out = pose_out

        if pose_out:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
            nn.init.kaiming_uniform_(self.W1.data)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)

    def forward(self, a, pose):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        b, _, h, w = a.shape

        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        if hasattr(self, 'W1'):
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)

        # [b, kkA, l]
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1,2)

        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1,2)

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        if hasattr(self, 'W1'):
            pose_out = pose_out.view(b, -1, oh, ow)
        else:
            pose_out = None

        return a_out, pose_out


