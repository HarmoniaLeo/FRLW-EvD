#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2]), int(in_channels[1]), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[1]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[0]), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0]), int(in_channels[0]), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[1]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[2]),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        #features = [out_features[f] for f in self.in_features]
        #[x2, x1, x0] = features
        [x2, x1, x0] = out_features
        
        fpn_out0 = self.lateral_conv0(x0)  # 
        f_out0 = self.upsample(fpn_out0)  # 
        f_out0 = torch.cat([f_out0, x1], 1)  # 
        f_out0 = self.C3_p4(f_out0)  # 

        fpn_out1 = self.reduce_conv1(f_out0)  # 
        f_out1 = self.upsample(fpn_out1)  # 
        f_out1 = torch.cat([f_out1, x2], 1)  # 
        pan_out2 = self.C3_p3(f_out1)  # 

        p_out1 = self.bu_conv2(pan_out2)  # 
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 
        pan_out1 = self.C3_n3(p_out1)  # 

        p_out0 = self.bu_conv1(pan_out1)  # 
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 
        pan_out0 = self.C3_n4(p_out0)  # 

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs
