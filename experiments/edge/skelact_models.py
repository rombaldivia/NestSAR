"""Ta-CNN, HCN(-Baseline) and Two-Stream CNN backbones + CNN head, copied from Hikvision SkelAct
(https://github.com/hikvision-research/skelact, Apache-2.0) with the mmcv/mmaction registry and init code removed.
Inference-only: used to benchmark latency. Input [N, C=3, T, V=25, M=2] as in SkelAct.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__(); self.dims = dims

    def forward(self, x):
        return x.permute(self.dims).contiguous()


class ConvBN(nn.Sequential):
    def __init__(self, i, o, k, stride=1, padding=0):
        super().__init__(nn.Conv2d(i, o, k, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(o))


class HCNBlock(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=64, num_joints=25, with_bn=False):
        inter = out_channels // 2
        conv1 = ConvBN if with_bn else nn.Conv2d
        super().__init__(conv1(in_channels, out_channels, 1), nn.ReLU(),
                         nn.Conv2d(out_channels, inter, (3, 1), padding=(1, 0)), Permute((0, 3, 2, 1)),
                         nn.Conv2d(num_joints, inter, 3, padding=1), nn.MaxPool2d(2, stride=2),
                         nn.Conv2d(inter, out_channels, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.Dropout(p=0.5))


class HCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, clip_len=64, with_bn=False, reduce="flatten"):
        super().__init__()
        self.reduce = reduce
        self.net_l = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.net_m = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.ReLU(), nn.Dropout(0.5))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.ReLU())
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Sequential(nn.Linear(256 * 2 * clip_len // 16, 256), nn.ReLU(), nn.Dropout(0.5)) \
            if reduce == "flatten" else None

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, t, v)
        vel = torch.cat((x[:, :, :1] * 0, x[:, :, 1:] - x[:, :, :-1]), dim=2)
        out = torch.cat((self.net_l(x), self.net_m(vel)), dim=1)
        out = self.conv6(self.conv5(out))
        if self.reduce == "mean":
            out = out.mean(dim=2)
        out = self.drop6(out.view(n, m, -1).max(dim=1)[0])
        return self.fc7(out) if self.fc7 is not None else out


class SELayer(nn.Module):        # torchvision SqueezeExcitation (avgpool -> fc1 -> ReLU -> fc2 -> sigmoid -> scale), no bias
    def __init__(self, c, squeeze_factor=1, bias=False):
        super().__init__()
        s = c // squeeze_factor
        self.fc1 = nn.Conv2d(c, s, 1, bias=bias); self.fc2 = nn.Conv2d(s, c, 1, bias=bias)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))
        return x * w


class DualGroupConv(nn.Module):
    def __init__(self, i, o, k, stride, padding, groups, bias):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, k, stride=stride, padding=padding, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(i, o, 1, stride=stride, padding=0, groups=groups // 2, bias=bias)

    def forward(self, x):
        return F.relu(self.conv1(x)) + F.relu(self.conv2(x))


class CrossChannelFeatureAugment(nn.Module):
    def __init__(self, i, o, kernel_size=3, stride=1, padding=1, groups=10, squeeze_factor=1):
        super().__init__()
        inter = (o // groups) * groups
        self.map1 = nn.Conv2d(i, inter, 1, bias=False)
        self.attend = SELayer(inter, squeeze_factor, bias=False)
        self.group = DualGroupConv(inter, inter, kernel_size, stride, padding, groups, bias=False)
        self.map2 = nn.Conv2d(inter, o, 1, bias=False)

    def forward(self, x):
        return F.relu(self.map2(self.group(self.attend(F.relu(self.map1(x))))))


class TaCNNBlock(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=64, num_joints=25, groups=(10, 6), squeeze_factor=1):
        inter = out_channels // 2
        super().__init__(ConvBN(in_channels, out_channels, 1), nn.ReLU(),
                         CrossChannelFeatureAugment(out_channels, inter, (3, 1), 1, (1, 0), groups[0], squeeze_factor),
                         Permute((0, 3, 2, 1)),
                         CrossChannelFeatureAugment(num_joints, inter, 3, 1, 1, groups[1], squeeze_factor),
                         nn.MaxPool2d(2, stride=2), nn.Conv2d(inter, out_channels, 3, padding=1),
                         nn.MaxPool2d(2, stride=2), nn.Dropout(0.5))


class TaCNN(HCN):
    def __init__(self, in_channels=3, num_joints=25, groups=(10, 6), squeeze_factor=1):
        super().__init__(in_channels, num_joints, 64, True, "mean")
        self.net_l = TaCNNBlock(in_channels, 64, num_joints, groups, squeeze_factor)
        self.net_m = TaCNNBlock(in_channels, 64, num_joints, groups, squeeze_factor)


class SkeletonTransformer(nn.Module):
    def __init__(self, in_joints=25, out_joints=30):
        super().__init__()
        self.trans_mat = nn.Parameter(torch.empty(in_joints, out_joints)); nn.init.orthogonal_(self.trans_mat)

    def forward(self, x):
        n, c, t, v = x.size()
        return torch.matmul(x.reshape(-1, v), self.trans_mat).view(n, c, t, -1)


class TSBlock(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=64, in_joints=25, out_joints=30):
        super().__init__(SkeletonTransformer(in_joints, out_joints),
                         nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.Dropout(0.5))


class TSCNN(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, clip_len=32):
        super().__init__()
        self.net_l = TSBlock(in_channels, 64, num_joints, 30); self.net_m = TSBlock(in_channels, 64, num_joints, 30)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.PReLU(init=0.1), nn.Dropout(0.5))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.PReLU(init=0.1))
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Sequential(nn.Linear(256 * 2 * clip_len // 8, 256), nn.PReLU(init=0.1), nn.Dropout(0.5))
        self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU(init=0.1))

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, t, v)
        vel = torch.cat((x[:, :, :1] * 0, x[:, :, 1:] - x[:, :, :-1]), dim=2)
        out = torch.cat((self.net_l(x), self.net_m(vel)), dim=1)
        out = self.drop4(self.conv4(self.conv3(out)).view(n, m, -1).max(dim=1)[0])
        return self.fc6(self.fc5(out))


class WithHead(nn.Module):
    def __init__(self, backbone, in_channels, num_classes=120):
        super().__init__(); self.backbone = backbone; self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def build(name, num_classes=120):
    """Configs of skelact/configs/_base_/models/{tacnn,hcnb,hcn,tscnn}_60.py; returns (model, input shape)."""
    if name == "tacnn":
        return WithHead(TaCNN(), 512, num_classes), (1, 3, 64, 25, 2)
    if name == "hcnb":
        return WithHead(HCN(clip_len=64, with_bn=True, reduce="mean"), 512, num_classes), (1, 3, 64, 25, 2)
    if name == "hcn":
        return WithHead(HCN(clip_len=64, with_bn=False, reduce="flatten"), 256, num_classes), (1, 3, 64, 25, 2)
    if name == "tscnn":
        return WithHead(TSCNN(clip_len=32), 128, num_classes), (1, 3, 32, 25, 2)
    raise ValueError(name)
