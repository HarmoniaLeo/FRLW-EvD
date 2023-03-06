from cgitb import small
from fileinput import filename
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse
from src.io.box_filtering import filter_boxes_gen1, filter_boxes_large
from src.metrics.coco_eval import evaluate_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str, default="gen1")   # Prophesee gen1/gen4 Dataset
    parser.add_argument('-exp_name', type=str)  # Name of experiment to evaluate result

    args = parser.parse_args()
    mode = "test"

    tol = 4999

    if args.dataset == "gen1":
        result_path = "log/" + args.exp_name + "/summarise_stats.npz"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        percentiles1 = [0.0, 0.09472751189131885, 0.2538587115258659, 0.6169536673563197, 1.703355726917305, 1000]
    else:
        result_path = "log/" + args.exp_name + "/summarise_stats.npz"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        percentiles1 = [0.0, 0.061864120261698595, 0.47486729209948575, 1.4415784200310098, 4.20493449274388, 1000]
        
        
    bbox_file = result_path
    f_bbox = np.load(bbox_file)
    dts = f_bbox["dts"]
    file_names_dt = f_bbox["file_names"]
    densitys_dt = f_bbox["densitys"]

    result_path = "statistics_result"
    bbox_file = os.path.join(result_path,"gt_"+args.dataset+".npz")
    f_bbox = np.load(bbox_file)
    gts = f_bbox["gts"]
    file_names_gt = f_bbox["file_names"]
    densitys_gt = f_bbox["densitys"]

    results = []

    for i in range(0,len(percentiles1)-1):
        print(i,percentiles1[i],percentiles1[i+1])
        dt = []
        gt = []

        for i_file, file_name in enumerate(np.unique(file_names_gt)):

            dt_bbox = dts[(file_names_dt == file_name)&(densitys_dt >= percentiles1[i])&(densitys_dt < percentiles1[i+1])]
            gt_bbox = gts[(file_names_gt == file_name)&(densitys_gt >= percentiles1[i])&(densitys_gt < percentiles1[i+1])]

            dt.append(dt_bbox)
            gt.append(gt_bbox)

        gt_boxes_list = map(filter_boxes, gt)
        result_boxes_list = map(filter_boxes, dt)
        gt_boxes_list1 = []
        result_boxes_list1 = []
        for l1,l2 in zip(gt_boxes_list,result_boxes_list):
            if len(l1) > 0:
                gt_boxes_list1.append(l1)
                if len(l2) == 0:
                    result_boxes_list1.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
                else:
                    result_boxes_list1.append(l2)
        
        result = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = tol, classes=classes,height=shape[0],width=shape[1])
        results.append(result[0])
    print(results)