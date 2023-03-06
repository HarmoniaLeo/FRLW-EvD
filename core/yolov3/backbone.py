import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys


__all__ = ['darknet53']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=p, stride=s, dilation=d),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, k=1),
                Conv_BN_LeakyReLU(ch//2, ch, k=3, p=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(
        self, 
        in_channels=3,
        stem = None
    ):
        super(DarkNet_53, self).__init__()
        # stride = 2
        if stem is None:
            self.layer_1 = nn.Sequential(
                Conv_BN_LeakyReLU(in_channels, 32, k=3, p=1),
                Conv_BN_LeakyReLU(32, 64, k=3, p=1, s=2),
                resblock(64, nblocks=1)
            )
            self.bfm = False
        else:
            self.layer_1 = stem(in_channels, 64, ksize=3, act="silu")
            self.bfm = True
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, k=3, p=1, s=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, k=3, p=1, s=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, k=3, p=1, s=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, k=3, p=1, s=2),
            resblock(1024, nblocks=4)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        if self.bfm:
            c1 = self.layer_1(x)
        else:
            c1 = self.layer_1(x[...,0])
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return (c3, c4, c5)


def darknet53(pretrained=False, hr=False, **kwargs):
    """Constructs a darknet-53 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the hi-res darknet53-448 ...')
            model_path = path_to_dir + '/weights/darknet53/darknet53_hr_77.76.pth'
            model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
        else:
            print('Loading the darknet53 ...')
            model_path = path_to_dir + '/weights/darknet53/darknet53_75.42.pth'
            model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
    
    return model
