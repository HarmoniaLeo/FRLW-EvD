from itertools import count
import numpy as np
from sklearn import datasets
from src.io import npy_events_tools, dat_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)   # "train, val, test" level direcotory of the datasets, for data source
    parser.add_argument('-target_dir', type=str)    # "train, val, test" level direcotory of the datasets, for data output
    parser.add_argument('-min_event_count', type=int, default=800000)   # Minimum event count before an annotation timestamp
    parser.add_argument('-sampling_period', type=int, default=1000000)   # Sampling period in μs

    args = parser.parse_args()

    min_event_count = args.min_event_count
    events_window_abin = 10000
    event_volume_bins = 5
    events_window = events_window_abin * event_volume_bins
    events_window_total = int(50000 + 16667 * 17)
    raw_dir = args.raw_dir
    target_dir = args.target_dir
    sampling_period = args.sampling_period

    for mode in ["train", "val", "test"]:
        
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        target_root = os.path.join(target_dir, mode)
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(files):
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(root, file_name + '_bbox.npy')
            new_event_file = os.path.join(target_root, file_name + '_td.dat')
            new_bbox_file = os.path.join(target_root, file_name + '_bbox.npy')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            f_event_new = open(new_event_file, "wb")

            time_upperbound = -1e16
            count_upperbound = -1
            already = False
            sampling = False

            sampled_events = []
            sampled_bboxes = []
            for bbox_count,unique_time in enumerate(unique_ts):
                if unique_time <= 500000:
                    continue
                if (unique_time - time_upperbound < sampling_period):
                    continue
                else:
                    sampling_start_time = unique_time
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - min_event_count
                if start_count < 0:
                    start_count = 0
                f_event.seek_event(start_count)
                start_time = int(f_event.current_time)
                if (end_time - start_time) < events_window_total:
                    start_time = end_time - events_window_total
                else:
                    start_time = end_time - round((end_time - start_time - events_window)/events_window_abin) * events_window_abin - events_window

                if start_time > time_upperbound:
                    start_count = f_event.seek_time(start_time)
                    if (start_count is None) or (start_time < 0):
                        start_count = 0
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
                sampled_events.append(events)
                sampled_bboxes.append(dat_bbox[dat_bbox['t']==unique_time])

                time_upperbound = end_time
                count_upperbound = end_count
            dat_events_tools.write_event_buffer(f_event_new, np.concatenate(sampled_events))
            sampled_bboxes = np.concatenate(sampled_bboxes)
            mmp = np.lib.format.open_memmap(new_bbox_file, "w+", dtype, sampled_bboxes.shape)
            mmp[:] = sampled_bboxes[:]
            mmp.flush()
            pbar.update(1)
        pbar.close()