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


def taf_cuda(x, y, t, p, shape, volume_bins, past_volume):
    tick = time.time()
    H, W = shape

    img = torch.zeros((H * W * 2)).float().to(x.device)
    img.index_add_(0, p + 2 * x + 2 * W * y, torch.ones_like(x).float())
    t_img = torch.zeros((H * W * 2)).float().to(x.device)
    t_img.index_add_(0, p + 2 * x + 2 * W * y, t - 1)
    t_img = t_img/(img+1e-8)

    img = img.view(H, W, 2)
    t_img = t_img.view(H, W, 2)
    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    tick = time.time()
    forward = (img == 0)
    torch.cuda.synchronize()
    filter_time = time.time() - tick
    tick = time.time()
    old_ecd = past_volume
    if torch.all(forward):
        ecd = old_ecd
    else:
        ecd = t_img[:, :, :, None]
        ecd = torch.cat([old_ecd, ecd],dim=3)
        for i in range(1,ecd.shape[3])[::-1]:
            ecd[:,:,:,i-1] = ecd[:,:,:,i-1] - 1
            ecd[:,:,:,i] = torch.where(forward, ecd[:,:,:,i-1],ecd[:,:,:,i])
        if ecd.shape[3] > volume_bins:
            ecd = ecd[:,:,:,1:]
        else:
            ecd[:,:,:,0] = torch.where(forward, torch.zeros_like(forward).float() -6000, ecd[:,:,:,0])
    torch.cuda.synchronize()
    generate_encode_time = time.time() - tick

    ecd_viewed = ecd.permute(3, 2, 0, 1).contiguous().view(volume_bins * 2, H, W)

    #print(generate_volume_time, filter_time, generate_encode_time)
    return ecd_viewed, ecd, generate_encode_time + generate_volume_time

def generate_taf_cuda(events, shape, past_volume = None, volume_bins=5):
    x, y, t, p, z = events.unbind(-1)

    x, y, t, p = x.long(), y.long(), t.float(), p.long()
    
    histogram_ecd, past_volume, generate_time = taf_cuda(x, y, t, p, shape, volume_bins, past_volume)

    return histogram_ecd, past_volume, generate_time

def leaky_transform(ecd):
    
    ecd = ecd.clone()
    ecd = torch.log1p(-ecd)
    ecd = 1 - ecd / 8.7
    ecd = torch.where(ecd < 0, torch.zeros_like(ecd), ecd)
    ecd = ecd * 255
    return ecd

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

    min_event_count = 50000000
    if dataset == "gen4":
        shape = [720,1280]
        target_shape = [512, 640]
    elif dataset == "kitti":
        shape = [375,1242]
        target_shape = [192, 640]
    else:
        shape = [240,304]
        target_shape = [256, 320]
    events_window_abin = 10000  #Delta tau = 10ms
    event_volume_bins = 8
    events_window = events_window_abin * event_volume_bins

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        label_root = os.path.join(label_dir, mode)
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        total_length = 0

        for i_file, file_name in enumerate(files):
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)
            total_length += len(unique_ts)

        pbar = tqdm.tqdm(total=total_length, unit='File', unit_scale=True)

        if mode == "test":
            total_time = 0
            total_count = 0

        for i_file, file_name in enumerate(files):
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            time_upperbound = -1e16
            count_upperbound = -1
            already = False
            sampling = False

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - min_event_count
                if start_count < 0:
                    start_count = 0
                f_event.seek_event(start_count)
                start_time = int(f_event.current_time)
                if (end_time - start_time) < events_window:
                    start_time = end_time - events_window
                else:
                    start_time = end_time - round((end_time - start_time - events_window)/events_window_abin) * events_window_abin - events_window

                if start_time > time_upperbound:
                    start_count = f_event.seek_time(start_time)
                    if (start_count is None) or (start_time < 0):
                        start_count = 0
                    memory = None
                else:
                    start_count = count_upperbound
                    start_time = time_upperbound
                    end_time = round((end_time - start_time) / events_window_abin) * events_window_abin + start_time
                    if end_time > f_event.total_time():
                        end_time = f_event.total_time()
                    end_count = f_event.seek_time(end_time)
                    assert bbox_count > 0

                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()
                
                z = torch.zeros_like(events[:,0])

                bins = math.ceil((end_time - start_time) / events_window_abin)
                
                for i in range(bins):
                    z = torch.where((events[:,2] >= start_time + i * events_window_abin)&(events[:,2] <= start_time + (i + 1) * events_window_abin), torch.zeros_like(events[:,2])+i, z)
                events = torch.cat([events,z[:,None]], dim=1)

                if start_time > time_upperbound:
                    if target_shape[0] < shape[0]:
                        memory = torch.zeros((target_shape[0], target_shape[1], 2, event_volume_bins)).cuda() - 6000    #6000个10ms，即60s（视频流长度），经f(.)变换后就会变为接近0的数值，作为事件表征中的默认值
                    else:
                        memory = torch.zeros((shape[0], shape[1], 2, event_volume_bins)).cuda() - 6000

                for iter in range(bins):
                    events_ = events[events[...,4] == iter]
                    t_max = start_time + (iter + 1) * events_window_abin
                    t_min = start_time + iter * events_window_abin
                    events_[:,2] = (events_[:, 2] - t_min)/(t_max - t_min + 1e-8)
                    if target_shape[0] < shape[0]:
                        events_[:,0] = events_[:,0] * rw
                        events_[:,1] = events_[:,1] * rh
                        volume, memory, generate_time = generate_taf_cuda(events_, target_shape, memory, event_volume_bins)
                    else:
                        volume, memory, generate_time = generate_taf_cuda(events_, shape, memory, event_volume_bins)
                        volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]
                if mode == "test": 
                    total_time += generate_time
                    total_count += 1
                volume = volume.view(event_volume_bins, 2, target_shape[0], target_shape[1])
                volume = leaky_transform(volume)
                ecd = volume.cpu().numpy().copy()
                ecd = np.flip(ecd, axis = 0)
                if not os.path.exists(os.path.join(target_root,"bins{0}".format(int(event_volume_bins/2)))):
                    os.makedirs(os.path.join(target_root,"bins{0}".format(int(event_volume_bins/2))))
                ecd[:4].astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"bins{0}".format(int(event_volume_bins/2))),file_name+"_"+str(unique_time)+".npy")) 
                if not os.path.exists(os.path.join(target_root,"bins{0}".format(event_volume_bins))):
                    os.makedirs(os.path.join(target_root,"bins{0}".format(event_volume_bins)))
                ecd[4:].astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"bins{0}".format(event_volume_bins)),file_name+"_"+str(unique_time)+".npy")) 
                
                time_upperbound = end_time
                count_upperbound = end_count
                torch.cuda.empty_cache()
                pbar.update(1)
        pbar.close()
    print("Average Representation time: ", total_time / total_count)