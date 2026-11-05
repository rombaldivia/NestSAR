"""SGN (Zhang et al., CVPR 2020) with a width multiplier, faithful to microsoft/SGN model.py.

width=1.0 is the original model (channel sizes 64 / 128 / 256 / 512). Every channel size, including the
64-wide hidden layer of the embedding MLPs, is scaled by `width`, so the cost scales ~width^2.
The one-hot joint/segment indices are built for the actual batch size (the original fixed it at train time).
"""
import math

import torch
from torch import nn


class NormData(nn.Module):
    def __init__(self, dim, joints=25):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim * joints)

    def forward(self, x):
        bs, c, v, t = x.shape
        return self.bn(x.reshape(bs, c * v, t)).reshape(bs, c, v, t)


def conv1x1(a, b, bias=True):
    return nn.Conv2d(a, b, kernel_size=1, bias=bias)


class Embed(nn.Module):
    def __init__(self, dim, out, hidden, norm, bias=True):
        super().__init__()
        layers = [NormData(dim)] if norm else []
        layers += [conv1x1(dim, hidden, bias), nn.ReLU(), conv1x1(hidden, out, bias), nn.ReLU()]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class Local(nn.Module):
    def __init__(self, d1, d2, seg, bias=True):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
        self.cnn1 = nn.Conv2d(d1, d1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(d1)
        self.cnn2 = nn.Conv2d(d1, d2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(d2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.cnn1(self.maxpool(x)))))
        return self.relu(self.bn2(self.cnn2(x)))


class GCNSpa(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(fout)
        self.relu = nn.ReLU()
        self.w = conv1x1(fin, fout, bias=False)
        self.w1 = conv1x1(fin, fout, bias=bias)

    def forward(self, x1, g):
        x = g.matmul(x1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.relu(self.bn(self.w(x) + self.w1(x1)))


class ComputeG(nn.Module):
    def __init__(self, d1, d2, bias=True):
        super().__init__()
        self.g1 = conv1x1(d1, d2, bias)
        self.g2 = conv1x1(d1, d2, bias)

    def forward(self, x):
        g1 = self.g1(x).permute(0, 3, 2, 1)
        g2 = self.g2(x).permute(0, 3, 1, 2)
        return torch.softmax(g1.matmul(g2), dim=-1)


class SGN(nn.Module):
    def __init__(self, num_classes, seg=20, width=1.0, joints=25, bias=True):
        super().__init__()
        c = max(4, int(round(64 * width)))
        self.c, self.seg, self.joints = c, seg, joints
        d1 = 4 * c
        self.tem_embed = Embed(seg, d1, c, norm=False, bias=bias)
        self.spa_embed = Embed(joints, c, c, norm=False, bias=bias)
        self.joint_embed = Embed(3, c, c, norm=True, bias=bias)
        self.dif_embed = Embed(3, c, c, norm=True, bias=bias)
        self.cnn = Local(d1, 2 * d1, seg, bias)
        self.compute_g1 = ComputeG(d1 // 2, d1, bias)
        self.gcn1 = GCNSpa(d1 // 2, d1 // 2, bias)
        self.gcn2 = GCNSpa(d1 // 2, d1, bias)
        self.gcn3 = GCNSpa(d1, d1, bias)
        self.fc = nn.Linear(2 * d1, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
        for g in (self.gcn1, self.gcn2, self.gcn3):
            nn.init.constant_(g.w.weight, 0)
        self.register_buffer("spa_onehot", torch.eye(joints).reshape(1, joints, joints, 1).expand(1, joints, joints, seg).clone(),
                             persistent=False)   # [1, V(channels), V, T]
        self.register_buffer("tem_onehot", torch.eye(seg).reshape(1, seg, 1, seg).expand(1, seg, joints, seg).clone(),
                             persistent=False)   # [1, T(channels), V, T]

    def forward(self, x):                        # x [B, seg, 75]
        bs, step, dim = x.shape
        x = x.reshape(bs, step, dim // 3, 3).permute(0, 3, 2, 1).contiguous()   # [B, 3, V, T]
        dif = torch.cat([torch.zeros_like(x[..., :1]), x[..., 1:] - x[..., :-1]], dim=-1)
        dy = self.joint_embed(x) + self.dif_embed(dif)
        spa = self.spa_embed(self.spa_onehot).expand(bs, -1, -1, -1)
        tem = self.tem_embed(self.tem_onehot).expand(bs, -1, -1, -1)
        h = torch.cat([dy, spa], 1)
        g = self.compute_g1(h)
        h = self.gcn3(self.gcn2(self.gcn1(h, g), g), g)
        h = self.cnn(h + tem)
        return self.fc(torch.flatten(torch.amax(h, dim=(2, 3), keepdim=True), 1))


def strict_mflops(model, seg=20):
    """multiply + add = 2 FLOPs, one forward of ONE sampled sequence (SGN feeds one 25-joint sequence per clip;
    its test protocol averages 5 random samplings = 5 forwards)."""
    from torch.utils.flop_counter import FlopCounterMode
    model = model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.randn(1, seg, 75))
    return fc.get_total_flops() / 1e6
