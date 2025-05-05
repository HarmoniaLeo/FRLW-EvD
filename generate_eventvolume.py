from pyexpat import features
from tkinter import S
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import torch
import time
import math
import argparse


def generate_agile_event_volume_cuda(events, shape, events_window = 50000, volume_bins=5):
    tick = time.time()
    H, W = shape

    x, y, t, p = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()

    t_star = (volume_bins * t.float())[:,None,None]
    channels = volume_bins


    adder = torch.stack([torch.arange(channels),torch.arange(channels)],dim = 1).to(x.device)[None,:,:] + 1   #1, 2, 2
    adder = (1 - torch.abs(adder-t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], channels * 2) #n, 4

    img = torch.zeros((H * W, volume_bins * 2)).float().to(x.device)
    img.index_add_(0, x + W * y, adder)
    img = img.view(H * W, volume_bins, 2)

    img_viewed = img.view((H, W, img.shape[1] * 2)).permute(2, 0, 1).contiguous()

    img_viewed = img_viewed / 5 * 255

    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    return img_viewed, generate_volume_time

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)

    features = dense_tensor[non_zero_indices[0],non_zero_indices[1],non_zero_indices[2]]

    return np.stack(non_zero_indices), features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)   # "train, val, test" level direcotory of the datasets, for data source
    parser.add_argument('-label_dir', type=str) # "train, val, test" level direcotory of the datasets, for reading annotations
    parser.add_argument('-target_dir', type=str)    # Data output directory
    parser.add_argument('-dataset', type=str, default="gen1")   # Perform experiment on Prophesee gen1/gen4 dataset

    args = parser.parse_args()
    raw_dir = args.raw_dir
    label_dir = args.label_dir
    target_dir = args.target_dir
    dataset = args.dataset

    if dataset == "gen4":
        shape = [720,1280]
        target_shape = [512, 640]
    else:
        shape = [240,304]
        target_shape = [256, 320]

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

    time_windows = [250000, 500000, 1000000]  # Delta tau = 50 ms, 100 ms, 200 ms
    event_volume_bins = 5

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        label_root = os.path.join(label_dir, mode)
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue

        files = [time_seq_name[:-7] for time_seq_name in files
                            if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        
        if mode == "test":
            total_time = [0 for i in time_windows]
            total_count = [0 for i in time_windows]

        for i_file, file_name in enumerate(files):
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    break
                start_time = int(end_time - np.max(time_windows))

                dat_event = f_event
                
                if start_time > 0:
                    dat_event.seek_time(start_time)
                    events = dat_event.load_delta_t(end_time - start_time)
                else:
                    dat_event.seek_time(0)
                    events = dat_event.load_delta_t(end_time)

                del dat_event
                events_ = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()
                events_ = events_[-10000000:]

                for i, time_window in enumerate(time_windows):
                    events = events_[events_[:,2] > end_time - time_window]

                    events[:,2] = (events[:,2] - (end_time - time_window)) / time_window

                    if target_shape[0] < shape[0]:
                        events[:,0] = events[:,0] * rw
                        events[:,1] = events[:,1] * rh
                        volume, generate_time = generate_agile_event_volume_cuda(events, target_shape, time_window, event_volume_bins)
                    else:
                        volume, generate_time = generate_agile_event_volume_cuda(events, shape, time_window, event_volume_bins)
                        volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]

                    if mode == "test":
                        total_time[i] += generate_time
                        total_count[i] += 1

                    volume = volume.cpu().numpy()
                    volume = np.where(volume > 255, 255, volume)
                    volume = volume.astype(np.uint8)

                    target_root = os.path.join(target_dir, "EventVolume{0}".format(time_window))
                    if not os.path.exists(target_root):
                        os.makedirs(target_root)

                    save_dir = os.path.join(target_root,mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    volume.tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))

                torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()
    if mode == "test":
        print("Average Representation time: ")
        for i,events_window in enumerate(time_windows):
            print(time_windows, total_time[i] / total_count[i])
