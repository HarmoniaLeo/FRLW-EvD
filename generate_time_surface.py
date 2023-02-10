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
    parser.add_argument('-raw_dir', type=str)   #数据集到train, val, test这一级的目录，作为源数据
    parser.add_argument('-label_dir', type=str) #数据集到train, val, test这一级的目录，用于读取标签
    parser.add_argument('-target_dir', type=str)    #输出数据的目标目录
    parser.add_argument('-dataset', type=str, default="gen4")   #prophesee gen1/gen4数据集
    #lamdas = [0.00001, 0.000005, 0.0000025, 0.000001]
    lamdas = [0.00001]  #lamda的数值列表

    args = parser.parse_args()
    raw_dir = args.raw_dir
    label_dir = args.label_dir
    target_dir = args.target_dir
    dataset = args.dataset

    if dataset == "gen4":
        # min_event_count = 800000
        shape = [720,1280]
        target_shape = [512, 640]
    elif dataset == "kitti":
        # min_event_count = 800000
        shape = [375,1242]
        target_shape = [192, 640]
    else:
        # min_event_count = 200000
        shape = [240,304]
        target_shape = [256, 320]
    events_window = 5000000
    time_window = 5541263 #554126,2216505,5541263

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
        # Remove duplicates (.npy and .dat)
        # files = files[int(2*len(files)/3):]
        #files = files[int(len(files)/3):]
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        total_time = 0
        total_count = 0

        for i_file, file_name in enumerate(files):
            # if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
            #     continue
            # if not file_name == "moorea_2019-06-26_test_02_000_1708500000_1768500000":
            #     continue
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            #h5 = h5py.File(volume_save_path, "w")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            #min_event_count = f_event.event_count()
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

                events_ = events[events[:,2] > end_time - time_window].clone()

                if target_shape[0] < shape[0]:
                    events_[:,0] = events_[:,0] * rw
                    events_[:,1] = events_[:,1] * rh
                    volume, memory, generate_time = generate_leaky_cuda(events_, target_shape, lamdas, memory, unique_time)
                else:
                    volume, memory, generate_time = generate_leaky_cuda(events_, shape, lamdas, memory, unique_time)
                    volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]

                volume = volume.view(len(lamdas), 2, target_shape[0], target_shape[1])

                total_time += generate_time
                total_count += 1
                #print(total_time / total_count)

                for j,i in enumerate(lamdas):
                    save_dir = os.path.join(target_dir,"leaky{0}".format(i))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_dir = os.path.join(save_dir, mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    ecd = volume[j].cpu().numpy().copy()
                    ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                
                # events_[:,2] = (events_[:,2] - (end_time - time_window)) / time_window
                # events = events_
                # torch.cuda.empty_cache()

                # if target_shape[0] < shape[0]:
                #     events[:,0] = events[:,0] * rw
                #     events[:,1] = events[:,1] * rh
                #     volume = generate_agile_event_volume_cuda(events, target_shape, time_window, 5)
                # else:
                #     volume = generate_agile_event_volume_cuda(events, shape, time_window, 5)
                #     volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]

                # volume = volume.cpu().numpy()
                # volume = np.where(volume > 255, 255, volume)
                # volume = volume.astype(np.uint8)

                # target_root_volume = os.path.join(target_dir_volume, "long{0}".format(time_window))
                # if not os.path.exists(target_root_volume):
                #     os.makedirs(target_root_volume)

                # save_dir = os.path.join(target_root_volume,mode)
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                
                # volume.tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                    
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()