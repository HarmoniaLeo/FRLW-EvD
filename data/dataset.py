#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from enum import unique
import os

import h5py
import cv2
import numpy as np
from numpy.lib import recfunctions as rfn
#import event_representations as er
from .prophesee import npy_events_tools
from .prophesee import psee_loader
from .utils import xyxy2cxcywh

import tqdm
import random
import time
import math

import torch

class propheseeDataset:
    def __init__(
        self,
        bbox_dir,
        data_dir,
        dataset="gen1",
        input_img_size = [256, 320],
        img_size = [256, 320],
        time_channels = 5,
        infer_time = 10000,
        train_memory_steps = 1,
        mode = "train",
        augment = True,
        clipping = False
    ):
        self.mode = mode
        self.augment = augment
        #self.augment = False

        file_dir = os.path.join(bbox_dir, self.mode)
        self.files = os.listdir(file_dir)
        # Remove duplicates (.npy and .dat)
        self.files = [time_seq_name[:-9] for time_seq_name in self.files
                      if time_seq_name[-3:] == 'npy']

        self.root = file_dir
        self.data_dir = data_dir

        if dataset == "gen1":
            self.width = 304
            self.height = 240
            self.object_classes = ['Car', "Pedestrian"]
        elif dataset == "kitti":
            self.width = 1242
            self.height = 375
            self.object_classes = ['Car', "Pedestrian"]
        else:
            self.width = 1280
            self.height = 720
            self.object_classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        
        self.clipping = clipping
        
        self.dataset = dataset
        
        self.input_img_size = input_img_size
        self.img_size = img_size
        self.time_channels = time_channels
        
        self.infer_time = infer_time
        self.train_memory_steps = train_memory_steps

        self.sequence_end_t = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.file_name)

    def createAllBBoxDataset(self):
        file_names = []
        print('Building the Dataset')

        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        taf_root = os.path.join(self.data_dir, self.mode)

        for i_file, file_name in enumerate(self.files):

            # if file_name in files_skip:
            #     continue

            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            # if self.dataset == "gen4":
            #     unique_ts = unique_ts[::2]
            
            for bbox_count,unique_time in enumerate(unique_ts):
                # if unique_time <= 500000:
                #     continue
                event_file = os.path.join(taf_root, file_name+ "_" + str(unique_time) + '.npy')
                if os.path.exists(event_file):
                    self.sequence_end_t.append(unique_time)
                    file_names.append(file_name)
            pbar.update(1)
        pbar.close()

        self.file_name = file_names
    
    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        bbox_file = os.path.join(self.root, self.file_name[idx] + '_bbox.npy')

        rh_ori = self.input_img_size[0] / self.height
        rw_ori = self.input_img_size[1] / self.width

        ng_augment = True

        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)

        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        indices = (dat_bbox['t'] == self.sequence_end_t[idx])
        #indices = (dat_bbox['t'] == self.sequence_end_t[idx])
        bboxes = dat_bbox[indices]

        unique_ts, unique_indices = np.unique(bboxes['t'], return_index=True)

        count = 0
        while ng_augment:
            
            if self.augment and (random.random() < 0.5):
                sr = random.uniform(1.0, 1.5)
            else:
                sr = 1.0

            if self.augment and (random.random() < 0.5):
                flip = True
            else:
                flip = False
            
            rh = sr * rh_ori
            rw = sr * rw_ori

            if sr < 1.0:
                cx = int(random.uniform(0, int(self.input_img_size[1] - sr * self.input_img_size[1])))
                cy = int(random.uniform(0, int(self.input_img_size[0] - sr * self.input_img_size[0])))
            if sr > 1.0:
                cx = int(random.uniform(int(self.input_img_size[1] - sr * self.input_img_size[1]), 0))
                cy = int(random.uniform(int(self.input_img_size[0] - sr * self.input_img_size[0]), 0))
            else:
                cx = 0
                cy = 0

            # Required Information ['x', 'y', 'w', 'h', 'class_id']
            np_bbox = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5, 0, 6, 7]]
            
            np_bbox = np.stack([np_bbox[:,0] * rw + cx, np_bbox[:,1] * rh + cy, (np_bbox[:,0] + np_bbox[:,2]) * rw + cx, (np_bbox[:,1] + np_bbox[:,3]) * rh + cy, np_bbox[:,4], np_bbox[:,5], np_bbox[:,6], np_bbox[:,7]],axis=-1)

            if self.dataset == "gen4":
                if self.augment:
                    np.clip(np_bbox[:, 0], 0, self.input_img_size[1], out=np_bbox[:, 0])
                    np.clip(np_bbox[:, 1], 0, self.input_img_size[0], out=np_bbox[:, 1])
                    np.clip(np_bbox[:, 2], 0, self.input_img_size[1], out=np_bbox[:, 2])
                    np.clip(np_bbox[:, 3], 0, self.input_img_size[0], out=np_bbox[:, 3])
                    np_bbox = np_bbox[(np_bbox[:, 2] - np_bbox[:, 0] > 5)&(np_bbox[:, 3] - np_bbox[:, 1] > 5)]
            else:
                if self.augment:
                    x2_overzero = (np_bbox[:, 2] > 10)
                    y2_overzero = (np_bbox[:, 3] > 10)
                    x1_underbound = (np_bbox[:, 0] < self.input_img_size[1] - 10)
                    y1_underbound = (np_bbox[:, 1] < self.input_img_size[0] - 10)
                    np_bbox = np_bbox[x2_overzero & x1_underbound & y1_underbound & y2_overzero]
            for t in unique_ts:
                ng_augment = (len(np_bbox[np_bbox[:, 5]==t]) == 0)
                if ng_augment:
                    break
            count += 1
            if count > 100:
                np_bbox = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5, 0, 6, 7]]
                np_bbox = np.stack([np_bbox[:,0] * rw_ori, np_bbox[:,1] * rh_ori, (np_bbox[:,0] + np_bbox[:,2]) * rw_ori, (np_bbox[:,1] + np_bbox[:,3]) * rh_ori, np_bbox[:,4], np_bbox[:,5], np_bbox[:,6], np_bbox[:,7]],axis=-1)
                rh = rh_ori
                rw = rw_ori
                break
        
        if (self.mode == "train" and self.clipping) or (self.dataset == "gen4"):
            np.clip(np_bbox[:, 0], 0, self.input_img_size[1], out=np_bbox[:, 0])
            np.clip(np_bbox[:, 1], 0, self.input_img_size[0], out=np_bbox[:, 1])
            np.clip(np_bbox[:, 2], 0, self.input_img_size[1], out=np_bbox[:, 2])
            np.clip(np_bbox[:, 3], 0, self.input_img_size[0], out=np_bbox[:, 3])

        boxes = np_bbox[:, :4].copy()
        labels = np_bbox[:, 4:].copy()

        if flip:
            boxes[:, 0::2] = self.input_img_size[1] - boxes[:, 2::-2] - 1
        
        boxes = xyxy2cxcywh(boxes)

        max_labels = 80

        if self.mode == "train":
            targets_t = np.hstack((labels[:,0:1],boxes))
        else:
            targets_t = np.hstack((boxes,labels))
        
        padded_labels = np.zeros((max_labels, targets_t.shape[1]),dtype = float)
        padded_labels[range(len(targets_t))] = targets_t

        img = self.load_data(idx)

        img = torch.from_numpy(img)

        img = torch.nn.functional.interpolate(img[None,:,:,:], size = (int(self.input_img_size[0] * sr), int(self.input_img_size[1] * sr)), mode='nearest')[0]

        img = self.after_process(img)
        
        img = img / 255

        img = img[:, -cy:self.input_img_size[0]-cy, -cx:self.input_img_size[1]-cx]

        img = img.numpy()

        if flip:
            img = img[:, :, ::-1]

        #output cxcywh
        #events xytpz

        return img, padded_labels, self.file_name[idx], self.sequence_end_t[idx]
    
    def load_data(self, idx):
        data_root = os.path.join(self.data_dir, self.mode)
        timestamp = self.sequence_end_t[idx]

        ecd_file = os.path.join(os.path.join(data_root), self.file_name[idx]+ "_" + str(timestamp) + ".npy")
        volume = np.fromfile(ecd_file, dtype=np.uint8).reshape(int(2*self.time_channels), self.img_size[0], self.img_size[1]).astype(np.float32)
        volume = np.stack([volume.mean(0),volume.mean(0)])

        return volume

    def after_process(self, img):
        return img[:, :, :, None, None]


class propheseeTafDataset(propheseeDataset):
    def __init__(self, bbox_dir, data_dir, dataset="gen1", input_img_size=[256, 320], img_size=[256, 320],  infer_time=10000, event_volume_bins=5, mode="train", augment=True, clipping=False):
        super().__init__(bbox_dir, data_dir, dataset, input_img_size, img_size, event_volume_bins, infer_time, 1, mode, augment, clipping)
    
    def createAllBBoxDataset(self):
        file_names = []
        print('Building the Dataset')

        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        taf_root = os.path.join(self.data_dir, self.mode)
        taf_root = os.path.join(taf_root, "bins8")

        for i_file, file_name in enumerate(self.files):

            # if file_name in files_skip:
            #     continue

            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            # if self.dataset == "gen4":
            #     unique_ts = unique_ts[::2]
            
            for bbox_count,unique_time in enumerate(unique_ts):
                # if unique_time <= 500000:
                #     continue
                event_file = os.path.join(taf_root, file_name+ "_" + str(unique_time) + '.npy')
                if os.path.exists(event_file):
                    self.sequence_end_t.append(unique_time)
                    file_names.append(file_name)
            pbar.update(1)
        pbar.close()
        self.file_name = file_names
    
    def load_data(self, idx):
        data_root = os.path.join(self.data_dir, self.mode)
        timestamp = self.sequence_end_t[idx]
        
        if self.time_channels > 4:
            ecd_file = os.path.join(os.path.join(data_root,"bins{0}".format(int(self.time_channels // 2))), self.file_name[idx]+ "_" + str(timestamp) + ".npy")
            volume = np.fromfile(ecd_file, dtype=np.uint8).reshape(int(self.time_channels), self.img_size[0], self.img_size[1]).astype(np.float32)
            ecd_file2 = os.path.join(os.path.join(data_root,"bins{0}".format(int(self.time_channels))), self.file_name[idx]+ "_" + str(timestamp) + ".npy")
            volume2 = np.fromfile(ecd_file2, dtype=np.uint8).reshape(int(self.time_channels), self.img_size[0], self.img_size[1]).astype(np.float32)
            volume = np.concatenate([volume, volume2], 0)
        else:
            ecd_file = os.path.join(os.path.join(data_root,"bins{0}".format(int(self.time_channels))), self.file_name[idx]+ "_" + str(timestamp) + ".npy")
            volume = np.fromfile(ecd_file, dtype=np.uint8).reshape(int(self.time_channels * 2), self.img_size[0], self.img_size[1]).astype(np.float32)

        return volume