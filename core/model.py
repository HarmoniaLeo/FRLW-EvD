#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

import time

class model(nn.Module):

    def __init__(self, backbone, neck, memory, head):
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.memory = memory
        self.head = head
    

                
        
    def forward(self, xs, targets, filenames, timestamps, evaluator = None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.training:
            outputs_tol = None
        for i in range(xs.shape[-1]):
            start = time.time()
            if i < xs.shape[-1] - 1:
                if not (self.memory is None):
                    backbone_outs = self.backbone(xs[...,i])
                    backbone_outs = self.memory(backbone_outs)
                    #fpn_outs = self.neck(backbone_outs)
                    #fpn_outs = self.memory(fpn_outs)
                if (self.head.seq_nms) and not (self.training):
                    backbone_outs = self.backbone(xs[...,i])
                    fpn_outs = self.neck(backbone_outs)
                    outputs = self.head(fpn_outs)
            else:
                backbone_outs = self.backbone(xs[...,i])
                if not (self.memory is None):
                    backbone_outs = self.memory(backbone_outs)
                fpn_outs = self.neck(backbone_outs)
                # torch.cuda.synchronize()
                # infer_time = time.time() - start
                # print("total",infer_time)
                # if not (self.memory is None):
                #     fpn_outs = self.memory(fpn_outs)
                if self.training:
                    losses = self.head(
                        fpn_outs, targets, xs[...,i]
                    )
                    if outputs_tol is None:
                        outputs_tol = losses[0]
                    else:
                        outputs_tol = outputs_tol + losses[0]
                else:
                    outputs = self.head(fpn_outs)
                    torch.cuda.synchronize()
                    infer_time = time.time() - start
                    evaluator.add_result(outputs, timestamps, targets, filenames, infer_time, 0)
            
        if not (self.memory is None):
            self.memory.clean_memory()
        if self.head.seq_nms:
            self.head.clean_seqnms()
        if self.training:
            return outputs_tol
        else:
            evaluator.end_a_batch()
            return evaluator