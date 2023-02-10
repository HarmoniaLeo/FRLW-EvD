from pyexpat import features
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
    parser.add_argument('-raw_dir', type=str)   #数据集到train, val, test这一级的目录，作为源数据
    parser.add_argument('-target_dir', type=str) #输出数据的目标目录
    parser.add_argument('-dataset', type=str, default="gen1")   #prophesee gen1/gen4数据集
    parser.add_argument('-time_window', type=int, default=50000)    #时间窗口，单位为微秒
    parser.add_argument('-event_volume_bins', type=int, default=5)  #Event Volume的bins数


    args = parser.parse_args()
    raw_dir = args.raw_dir
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

    # event_volume_bins = [[5, 1, 1], [5, 1]]
    # time_windows = [50000, 300000]
    # time_steps = [[16, 1, 1], [1, 1]]
    # cats = [["ev", "ev", "ts"], ["ev", "ts"]]
    # time_step = 10000

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

    time_window = args.time_window
    #time_steps = [[1]]
    event_volume_bins = args.event_volume_bins
    time_step = 10000

    #target_dirs = [os.path.join(target_dir, cat) for cat in ["normal", "long", "short", "ts_short", "ts_long"]]
    #target_dirs = [os.path.join(target_dir, cat) for cat in ["normal"]]
    #target_dir = os.path.join(target_dir, "normal")


    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        root = file_dir
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue

        files = [time_seq_name[:-7] for time_seq_name in files
                            if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        target_root = os.path.join(target_dir, mode)
    
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        
        # Remove duplicates (.npy and .dat)
        # files = files[int(2*len(files)/3):]
        #files = files[int(len(files)/3):]
            

        for i_file, file_name in enumerate(files):
            # if not file_name == "moorea_2019-06-26_test_02_000_1708500000_1768500000":
            #     continue
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            for bbox_count,unique_time in enumerate(unique_ts):
                volume_save_path = os.path.join(target_root, file_name+"_"+str(unique_time)+".npy")
                if os.path.exists(volume_save_path):
                   continue
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    break
                start_time = end_time - time_window

                dat_event = f_event
                
                if start_time > 0:
                    dat_event.seek_time(start_time)
                    events = dat_event.load_delta_t(end_time - start_time)
                else:
                    dat_event.seek_time(0)
                    events = dat_event.load_delta_t(end_time)

                del dat_event
                events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()

                events[:,2] = (events[:,2] - start_time) / time_window

                if target_shape[0] < shape[0]:
                    events[:,0] = events[:,0] * rw
                    events[:,1] = events[:,1] * rh
                    #start = time.time()
                    volume = generate_agile_event_volume_cuda(events, target_shape, time_window, event_volume_bins)
                    # torch.cuda.synchronize()
                    # end = time.time()
                else:
                    #start = time.time()
                    volume = generate_agile_event_volume_cuda(events, shape, time_window, event_volume_bins)
                    # torch.cuda.synchronize()
                    # end = time.time()
                    volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]
                
                #print(end - start)

                volume = volume.cpu().numpy()
                volume = np.where(volume > 255, 255, volume)
                volume = volume.astype(np.uint8)
                locations, features = denseToSparse(volume)

                c, y, x = locations
                p = c%2
                c = (c/2).astype(int)

                volume = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10) + np.left_shift(c.astype(np.uint32), 19) + np.left_shift(p.astype(np.uint32), 22) + np.left_shift(features.astype(np.uint32), 23)

                x = np.bitwise_and(volume, 1023).astype(int)
                y = np.right_shift(np.bitwise_and(volume, 523264), 10).astype(int)
                c = np.right_shift(np.bitwise_and(volume, 3670016), 19).astype(int)
                p = np.right_shift(np.bitwise_and(volume, 4194304), 22).astype(int)
                features = np.right_shift(np.bitwise_and(volume, 2139095040), 23).astype(int)

                events = np.stack([x, y, c, p, features], axis=1)

                volume.tofile(volume_save_path)
                #features.tofile(volume_save_path_f)
                #np.savez(volume_save_path, locations = locations, features = features)
                #h5.create_dataset(str(unique_time)+"/locations", data=locations)
                #h5.create_dataset(str(unique_time)+"/features", data=features)
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()