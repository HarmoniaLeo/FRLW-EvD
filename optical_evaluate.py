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
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-tol', type = int, default=4999)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        result_path = "/home/lbd/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        #percentiles = [0.0, 0.023955019742690712, 0.042253373088964444, 0.07143436976739469, 0.101650021925867, 0.13346816647302767, 0.16817225179138567, 0.20777102878868156, 0.2576741573299752, 0.317873322207675, 0.39235074076386517, 0.4885132853135831, 0.613289039892126, 0.7680029836511075, 0.9726207443471735, 1.2624602141157863, 1.6633663775961032, 2.234325892848631, 3.051188316051193, 4.3239727771346255]
        #percentiles = [0.0, 0.022090551140021526, 0.03307005296535371, 0.05671614855808474, 0.08410216123034962, 0.11406982861849572, 0.14831617911666944, 0.18462005594938136, 0.22870882848611626, 0.2853853858727559, 0.3554077913283515, 0.44819659632660175, 0.5715129938105777, 0.7367699321420207, 0.9613864329222702, 1.2917300269541516, 1.7416549267996466, 2.4035640687750264, 3.302250419545379, 4.655760829519853, 1000]
        #percentiles = [0.0, 0.03307005296535371, 0.08410216123034962, 0.14831617911666944, 0.22870882848611626, 0.3554077913283515, 0.5715129938105777, 0.9613864329222702, 1.7416549267996466, 3.302250419545379, 1000]
        #[0.0, 0.022826836845540655, 0.037013073408681124, 0.06448076630071245, 0.09472751189131885, 0.12681692210188172, 0.16266472495121756, 0.20304601001680278, 0.2538587115258659, 0.3154112080140129, 0.39181042411402056, 0.4893193445564121, 0.6169536673563197, 0.7805525190114224, 0.9951987812240594, 1.2969357872594736, 1.703355726917305, 2.296600946458227, 3.121383262000021, 4.374163669844856]
        #percentiles = [0.0, 0.08410216123034962, 0.22870882848611626, 0.5715129938105777, 1.7416549267996466, 1000]
        #percentiles = [0.0, 0.101650021925867, 1.6633663775961032, 1000]
        percentiles2 = [0.0, 0.12681692210188172, 1.2969357872594736, 1000]
        percentiles1 = [0.0, 0.09472751189131885, 0.2538587115258659, 0.6169536673563197, 1.703355726917305, 1000]
    else:
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        #percentiles = [0.0, 0.13798373765648486, 0.6976278461290158, 2.025066711356914, 5.278082102864997, 1000]
        #percentiles = [0.0, 0.0, 0.0, 0.017224068593383286, 0.04395048958884435, 0.10717341445938622, 0.20183185315356414, 0.31609882595677286, 0.4509914337597868, 0.6113675070465179, 0.811929173028783, 1.0687768126032453, 1.4313008685647621, 1.9108096336945133, 2.5080834649774104, 3.238813250464127, 4.108326745357363, 5.203311921389142, 6.6210589062068905, 8.648134991471418]
        #percentiles = [0.0, 0.0, 0.0, 0.018915752892064257, 0.0489449954503074, 0.11323120410190246, 0.20295585924830123, 0.3099795391225571, 0.4390496015194756, 0.59493510328393, 0.78934969219668, 1.0402726076857434, 1.3904864304128026, 1.862398078728036, 2.463074497303443, 3.211974451696608, 4.130296275763464, 5.292283327513696, 6.754746824399112, 8.873209713168537]
        percentiles2 = [0.0, 0.13691002283099496, 3.2786494067846, 1000]
        #percentiles2 = [0.0, 0.0, 0.0, 0.024069758977451696, 0.061864120261698595, 0.13691002283099496, 0.23318884073192203, 0.34242255943500355, 0.47486729209948575, 0.6336573219372644, 0.8326946149540354, 1.0854665004882589, 1.4415784200310098, 1.9203229983640102, 2.529642175121489, 3.2786494067846, 4.20493449274388, 5.370377098119524, 6.799594222218903, 8.962206297968446]
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
            #dt_bbox = dts[(file_names_dt == file_name)]
            #gt_bbox = gts[(file_names_gt == file_name)]

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
        
        result = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
        results.append(result[0])
    print(results)
    for i in range(0,len(percentiles2)-1):
        print(i,percentiles2[i],percentiles2[i+1])
        dt = []
        gt = []

        for i_file, file_name in enumerate(np.unique(file_names_gt)):

            dt_bbox = dts[(file_names_dt == file_name)&(densitys_dt >= percentiles2[i])&(densitys_dt < percentiles2[i+1])]
            gt_bbox = gts[(file_names_gt == file_name)&(densitys_gt >= percentiles2[i])&(densitys_gt < percentiles2[i+1])]
            #dt_bbox = dts[(file_names_dt == file_name)]
            #gt_bbox = gts[(file_names_gt == file_name)]

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
        
        result = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
        results.append(result[0])
    print(results)