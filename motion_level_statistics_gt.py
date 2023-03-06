from cgitb import small
from telnetlib import X3PAD
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import pandas as pd
import argparse

def nms(dets):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 1]  #xmin
    y1 = dets[:, 2]  #ymin
    x2 = dets[:, 3]  #xmax
    y2 = dets[:, 4]  #ymax

    areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
    order = np.arange(len(dets))                        # sort bounding boxes by decreasing order

    keep = []                                             # store the final bounding boxes
    while order.size > 0:
        i = order[0]                                      #the index of the bbox with highest confidence
        keep.append(i)                                    #save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= 0.1)[0]
        if len(inds) != len(ovr):
            keep.pop()
        order = order[inds + 1]

    return keep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)   # "train, val, test" level direcotory of the datasets, for reading annotations
    parser.add_argument('-dataset', type=str)   # Prophesee gen1/gen4 Dataset

    args = parser.parse_args()
    mode = "test"

    raw_dir = args.raw_dir

    if args.dataset == "gen1":
        shape = [240,304]
    else:
        shape = [720,1280]
    
    result_path = "statistics_result"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    file_dir = os.path.join(raw_dir, mode)
    root = file_dir
    files = os.listdir(file_dir)

    # Remove duplicates (.npy and .dat)
    files = [time_seq_name[:-7] for time_seq_name in files
                    if time_seq_name[-3:] == 'dat']

    pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

    file_names = []
    gt = []
    densitys = []

    for i_file, file_name in enumerate(files):
        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        dat_bbox = rfn.structured_to_unstructured(dat_bbox)

        unique_ts, unique_indices = np.unique(dat_bbox[:,0], return_index=True)

        for bbox_count,unique_time in enumerate(unique_ts):

            gt_trans = dat_bbox[dat_bbox[:,0] == unique_time]

            flow = np.load(os.path.join("optical_flow_buffer",file_name + "_{0}.npy".format(int(unique_time))))

            gt_nms = gt_trans.copy()
            gt_nms[:,3] = gt_trans[:,3] + gt_trans[:,1]
            gt_nms[:,4] = gt_trans[:,4] + gt_trans[:,2]

            gt_trans = gt_trans[nms(gt_nms)]

            for j in range(len(gt_trans)):
                x1, y1, x2, y2 = gt_trans[j,1], gt_trans[j,2], gt_trans[j,3] + gt_trans[j,1], gt_trans[j,4] + gt_trans[j,2]

                if x1 >= shape[1]:
                    x1 = shape[1] - 1
                if x1 < 0:
                    x1 = 0
                if x2 >= shape[1]:
                    x2 = shape[1] - 1
                if x2 < 0:
                    x2 = 0
                
                if y1 >= shape[0]:
                    y1 = shape[0] - 1
                if y1 < 0:
                    y1 = 0
                if y2 >= shape[0]:
                    y2 = shape[0] - 1
                if y2 < 0:
                    y2 = 0

                file_names.append(file_name)
                gt_trans[j,1] = x1
                gt_trans[j,2] = y1
                gt_trans[j,3] = x2 - x1
                gt_trans[j,4] = y2 - y1
                gt.append(gt_trans[j])

                density = np.sum(np.sqrt(flow[int(y1):int(y2),int(x1):int(x2),0]**2 + flow[int(y1):int(y2),int(x1):int(x2),1]**2))/(int(y2 - y1)*int(x2 - x1) + 1e-8)
                densitys.append(density)

        pbar.update(1)
    pbar.close()
    csv_path = os.path.join(result_path,"gt_"+args.dataset+".npz")
    print([np.quantile(densitys,q/100) for q in range(0,100,5)])
    np.savez(csv_path,
        file_names = file_names,
        gts = gt,
        densitys = densitys)