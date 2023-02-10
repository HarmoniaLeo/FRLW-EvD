import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)

class YOLOv3FPN(nn.Module):
    def __init__(
        self
    ):
        super(YOLOv3FPN, self).__init__()
        
        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )


    def forward(self, x, target=None):
        # backbone
        c3, c4, c5 = x

        # FPN, 多尺度特征融合
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)

        return p3, p4, p5