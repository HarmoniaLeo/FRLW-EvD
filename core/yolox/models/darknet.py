#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn
import torch
from torch.autograd import Variable

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from core.Others.Temporal_Active_Focus import Temporal_Active_Focus_3D, Temporal_Active_Focus_corr, Temporal_Active_Focus_swin

import time

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        shape,
        stem = Focus, 
        in_channels=3,
        stem_out_channels=64,
        out_channels = [256, 512, 1024],
        out_features=("dark3", "dark4", "dark5"),
        act="silu",
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        '''self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1,act=act),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,act=act),
        )'''
        
        self.stem = stem(in_channels, stem_out_channels, ksize=3, act=act)
        base_channels = stem_out_channels

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.

        # self.dark1 = nn.Sequential(
        #     *self.make_group_layer(base_channels, base_channels * 2, num_blocks[0], stride=2,act=act)
        # )

        self.dark2 = nn.Sequential(
            *self.make_group_layer(base_channels * 1, base_channels * 2, num_blocks[0], stride=2,act=act)
        )

        self.dark3 = nn.Sequential(
            *self.make_group_layer(base_channels * 2, out_channels[0], num_blocks[1], stride=2,act=act)
        )

        self.dark4 = nn.Sequential(
            *self.make_group_layer(out_channels[0], out_channels[1], num_blocks[2], stride=2,act=act)
        )

        self.dark5 = nn.Sequential(
            *self.make_group_layer(out_channels[1], out_channels[2], num_blocks[3], stride=2,act=act),
            *self.make_spp_block([out_channels[2], out_channels[2]], base_channels * 4,act=act),
        )

        self.shape = shape
        

    def make_group_layer(self, in_channels, out_channels, num_blocks, stride, act = "silu"):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, out_channels, ksize=3, stride=stride,act=act),
            *[(ResLayer(out_channels,act=act)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters,act="silu"):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act=act),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation=act,
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act=act),
            ]
        )
        return m

    def forward(self, input):
        outputs = {}
        x = self.stem(input)    #64, 128, 160, 0.9490G (+0.0679G)
        outputs["stem"] = x
        #x = self.dark1(x)
        #outputs["dark1"] = x
        x = self.dark2(x)    #128, 64, 80, 1.5584G
        outputs["dark2"] = x
        x = self.dark3(x)    #256, 32, 40, 2.1853G
        outputs["dark3"] = x
        x = self.dark4(x)    #256, 16, 20, 0.6779G
        outputs["dark4"] = x
        x = self.dark5(x)    #256, 8, 10, 0.1281G 0.1114G
        outputs["dark5"] = x
        #4.7300G
        #5.2454G 1.0915G 0.1807G
        #16.8602G
        return [outputs[k] for k in self.out_features]
        #return {k: v for k, v in outputs.items() if k in self.out_features}

class SEAttention(nn.Module):

    def __init__(self, channel=512,out_channel=512,reduction=16, act = "gelu"):
        super().__init__()
        # self.conv = BaseConv(channel, channel, 1, 1, act = act)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv2 = BaseConv(channel, out_channel, 1, 1, act = act)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #return self.conv2(x)
        x = self.conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.conv2(x * y.expand_as(x))


class SwinDarknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        shape,
        stem = Focus, 
        in_channels=3,
        stem_out_channels=64,
        out_channels = [256, 512, 1024],
        out_features=("dark3", "dark4", "dark5"),
        act="silu",
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        '''self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1,act=act),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,act=act),
        )'''
        
        self.stem = stem(in_channels, stem_out_channels, ksize=3, act=act)
        base_channels = stem_out_channels

        self.stem2 = Temporal_Active_Focus_3D(in_channels, base_channels, act = act)
        #self.stem2 = Temporal_Active_Focus_swin(in_channels, base_channels, act = act)

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.

        # self.dark1 = nn.Sequential(
        #     *self.make_group_layer(base_channels, base_channels * 2, num_blocks[0], stride=2,act=act)
        # )

        self.se = SEAttention(base_channels * 2, base_channels * 2, 4, act = act)

        self.dark2 = nn.Sequential(
            *self.make_group_layer(base_channels * 2, base_channels, num_blocks[0], stride=2,act=act)
        )

        self.dark3 = nn.Sequential(
            *self.make_group_layer(base_channels, out_channels[0], num_blocks[1], stride=2,act=act)
        )

        self.dark4 = nn.Sequential(
            *self.make_group_layer(out_channels[0], out_channels[1], num_blocks[2], stride=2,act=act)
        )

        self.dark5 = nn.Sequential(
            *self.make_group_layer(out_channels[1], out_channels[2], num_blocks[3], stride=2,act=act),
            *self.make_spp_block([out_channels[2], out_channels[2]], base_channels * 4,act=act),
        )

        self.shape = shape
        

    def make_group_layer(self, in_channels, out_channels, num_blocks, stride, act = "silu"):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, out_channels, ksize=3, stride=stride,act=act),
            *[(ResLayer(out_channels,act=act)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters,act="silu"):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act=act),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation=act,
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act=act),
            ]
        )
        return m

    def forward(self, input):
        outputs = {}
        x = self.stem(input)    #64, 128, 160, 0.9490G (+0.0679G)
        outputs["stem"] = x
        x2 = self.stem2(input)
        x = torch.cat([x, x2], 1)
        x = self.se(x)
        x = self.dark2(x)    #128, 64, 80, 1.5584G
        outputs["dark2"] = x
        #x = self.dark1(x)
        #outputs["dark1"] = x
        x = self.dark3(x)    #256, 32, 40, 2.1853G
        outputs["dark3"] = x
        x = self.dark4(x)    #256, 16, 20, 0.6779G
        outputs["dark4"] = x
        x = self.dark5(x)    #256, 8, 10, 0.1281G 0.1114G
        outputs["dark5"] = x
        #4.7300G
        #5.2454G 1.0915G 0.1807G
        #16.8602G
        return [outputs[k] for k in self.out_features]
        #return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPDarknet(nn.Module):
    def __init__(
        self,
        in_channel,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        stem = Focus
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = stem(in_channel, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return [outputs[k] for k in self.out_features]
        #return {k: v for k, v in outputs.items() if k in self.out_features}
