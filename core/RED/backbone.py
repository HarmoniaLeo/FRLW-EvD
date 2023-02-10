from turtle import forward
import torch.nn as nn
import math
from core.Others.memory_blocks import ConvLSTMCell, memoryModel, makeMemoryBlocks

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual

        return res


class SEResNet(memoryModel):

    def __init__(self, in_channels):
        super().__init__(None)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = Bottleneck(32, 64, 2)
        self.layer2 = Bottleneck(64, 64, 2)
        self.layer3 = Bottleneck(64, 128, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x[...,0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    #32, 240, 320, 2.418G

        x = self.layer1(x)    #64, 120, 160, 5.777G
        x = self.layer2(x)    #64, 60, 80, 2.172G
        x = self.layer3(x)    #128, 30, 40, 1.440G

        return x

class MemoryLayers(memoryModel):
    def __init__(self):
        super().__init__(makeMemoryBlocks(ConvLSTMCell, [3, 3, 3, 3, 3], [128, 256, 256, 256, 256], [256, 256, 256, 256, 256], [2, 2, 2, 2, 2]))
        # self.conv1 = nn.Conv2d(128, 256, kernel_size=1, bias=False, stride=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.conv2 = nn.Conv2d(256, 256, kernel_size=3, bias=False, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, bias=False, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, bias=False, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, bias=False, stride=2, padding=1)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        x = self.lstms[0](x)
        outputs.append(x)    #15, 20, 2.1245G 0.0581G
        x = self.lstms[1](x)
        outputs.append(x)    #7, 10, 0.7553G 0.0136G
        x = self.lstms[2](x)
        outputs.append(x)    #4, 5, 0.1888G 0.0039G
        x = self.lstms[3](x)
        outputs.append(x)    #2, 3, 0.0566G 0.0012G
        x = self.lstms[4](x)
        outputs.append(x)    #1, 1, 0.0189G 0.0002G
        #15.0281G
        return outputs