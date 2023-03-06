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

def generate_agile_event_volume_cuda(events, shape, events_window = 50000, volume_bins=5):
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

    # print(torch.quantile(img_viewed[img_viewed>0],0.95))

    img_viewed = img_viewed / 5 * 255

    return img_viewed

def taf_cuda(x, y, t, p, shape, lamdas, memory, now):
    tick = time.time()
    H, W = shape

    t_img = torch.zeros((2, H, W)).float().to(x.device) + now - 5000000
    t_img.index_put_(indices= [p, y, x], values= t)

    if not memory is None:
        t_img = torch.where(t_img>memory, t_img, memory)

    memory = t_img
    t_img = t_img - now

    t_imgs = []
    for lamda in lamdas:
        t_img_ = torch.exp(lamda * t_img)
        t_imgs.append(t_img_)
    ecd = torch.stack(t_imgs, 0)

    ecd_viewed = ecd.view(len(lamdas) * 2, H, W) * 255

    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    #print(generate_volume_time, filter_time, generate_encode_time)
    return ecd_viewed, memory, generate_volume_time

def generate_leaky_cuda(events, shape, lamdas, memory, now):
    events = events[(events[:,0]<shape[1])&(events[:,1]<shape[0])]

    x, y, t, p = events.unbind(-1)

    x, y, t, p = x.long(), y.long(), t.float(), p.long()
    
    histogram_ecd, memory, generate_volume_time = taf_cuda(x, y, t, p, shape, lamdas, memory, now)

    return histogram_ecd, memory, generate_volume_time

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
    elif dataset == "kitti":
        shape = [375,1242]
        target_shape = [192, 640]
    else:
        shape = [240,304]
        target_shape = [256, 320]

    lamdas = [0.00001, 0.0000025, 0.0000001]  #lambda = 0.00001, 0.0000025, 0.0000001
    time_window = [554126, 2216505, 5541263]
    
    events_window = 5000000

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

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
            total_time = [0 for i in time_window]
            total_count = [0 for i in time_window]

        for i_file, file_name in enumerate(files):
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            time_upper_bound = -100000000
            count_upper_bound = 0
            memory = None

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_time = int(f_event.current_time) 
                start_time = end_time - events_window
                if start_time < 0:
                    start_count = f_event.seek_time(0)
                else:
                    start_count = f_event.seek_time(start_time)
                
                if (start_count is None) or (start_time < 0):
                    start_count = 0
                
                if start_time <= time_upper_bound:
                    start_count = count_upper_bound
                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()

                time_upper_bound = unique_time
                count_upper_bound = end_count

                if mode == "test":
                    time_windows = time_window
                else:
                    time_windows = [max(time_window)]
                
                max_volume = None
                
                for i,tw in enumerate(time_windows):

                    events_ = events[events[:,2] > end_time - tw].clone()

                    if target_shape[0] < shape[0]:
                        events_[:,0] = events_[:,0] * rw
                        events_[:,1] = events_[:,1] * rh
                        volume, memory, generate_time = generate_leaky_cuda(events_, target_shape, lamdas, memory, unique_time)
                    else:
                        volume, memory, generate_time = generate_leaky_cuda(events_, shape, lamdas, memory, unique_time)
                        volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]

                    volume = volume.view(len(lamdas), 2, target_shape[0], target_shape[1])

                    if mode == "test":
                        total_time[i] += generate_time
                        total_count[i] += 1
                    
                    if tw == max(time_window):
                        max_volume = volume

                for j,i in enumerate(lamdas):
                    save_dir = os.path.join(target_dir,"leaky{0}".format(i))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_dir = os.path.join(save_dir, mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    ecd = max_volume[j].cpu().numpy().copy()
                    ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                        
                    torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()
    print("Average Representation time: ")
    for i,events_window in enumerate(time_windows):
        print(time_windows, total_time[i] / total_count[i])