from itertools import count
import numpy as np
from sklearn import datasets
from sqlalchemy import false
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import h5py
import pickle
import torch
import time
import math
import argparse
import torch.nn


def generate_eventframe(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
    coordinates u and v.
    """
    tick = time.time()
    H, W = shape
    x, y, t, p = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()

    img = torch.zeros((H * W * 2,)).float().to(x.device)

    img.index_add_(0, 2 * x + 2 * W * y + p, torch.zeros_like(x).float()+0.05)

    img = torch.where(img > 1, torch.ones_like(img).float(), img)

    histogram = img.view((H, W, 2)).permute(2, 0, 1).contiguous()

    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    return histogram * 255, generate_volume_time

def generate_frame(events, shape, events_window = 50000, volume_bins=5):
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

    return img_viewed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)   # "train, val, test" level direcotory of the datasets, for data source
    parser.add_argument('-label_dir', type=str) # "train, val, test" level direcotory of the datasets, for reading annotations
    parser.add_argument('-target_dir', type=str)    # Data output directory
    parser.add_argument('-dataset', type=str, default="gen4")   # Perform experiment on Prophesee gen1/gen4 dataset

    args = parser.parse_args()
    raw_dir = args.raw_dir
    label_dir = args.label_dir
    target_dir = args.target_dir
    dataset = args.dataset

    if dataset == "gen4":
        shape = [720,1280]
        target_shape = [512, 640]
        events_windows = [400000, 800000, 1200000]  # N = 400000, 800000, 1200000
    else:
        shape = [240,304]
        target_shape = [256, 320]
        events_windows = [50000, 100000, 200000]  # N = 50000, 100000, 200000
    

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

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
            total_time = [0 for i in events_windows]
            total_count = [0 for i in events_windows]

        for i_file, file_name in enumerate(files):

            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            count_upper_bound = -100000000
            memory = None

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = int(end_count - np.max(events_windows))
                if start_count < 0:
                    start_count = 0
                if start_count <= count_upper_bound:
                    start_count = count_upper_bound
                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()

                if not memory is None:
                    events = torch.cat([memory, events])

                memory = events[-np.max(events_windows):]
                count_upper_bound = end_count

                for i,events_window in enumerate(events_windows):
                
                    events_ = events[-events_window:].clone()
                    
                    if target_shape[0] < shape[0]:
                        events_[:,0] = events_[:,0] * rw
                        events_[:,1] = events_[:,1] * rh
                        volume, generate_time = generate_eventframe(events_, target_shape)
                    else:
                        volume, generate_time = generate_eventframe(events_, shape)
                        volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]
                    
                    if mode == "test":
                        total_time[i] += generate_time
                        total_count[i] += 1
                    

                    ecd_dir = os.path.join(target_dir,"EventCountImage{0}".format(events_window))
                    if not os.path.exists(ecd_dir):
                        os.makedirs(ecd_dir)
                    save_dir = os.path.join(ecd_dir, mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    ecd = volume.cpu().numpy().copy()
                    
                    ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                            
                torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()
    print("Average Representation time: ")
    for i,events_window in enumerate(events_windows):
        print(events_windows, total_time[i] / total_count[i])