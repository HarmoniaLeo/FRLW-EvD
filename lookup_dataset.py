import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import os
import cv2
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
sns.set_style("darkgrid")

def generate_event_volume(events,shape,ori_shape,C):

    x, y, c, p, features = events.T

    H, W = shape

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume_t = feature_map.reshape(C, H, W, 2)

    volume_t = volume_t.transpose(0,3,1,2).reshape(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])

    volume_t = torch.from_numpy(volume_t[None,:,:,:])
    volume_t = torch.nn.functional.interpolate(volume_t, size = ori_shape, mode='nearest')[0]
    return volume_t.numpy()


LABELMAP = ["car", "pedestrian"]

def draw_bboxes(img, boxes, dt, labelmap):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i][1]), int(boxes[i][2]))
        size = (int(boxes[i][3]), int(boxes[i][4]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes[i][-2]
        class_id = boxes[i][-3]
        class_name = labelmap[int(class_id)]
        color = colors[(dt+1) * 60]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        if dt:
            cv2.rectangle(img, (pt1[0], pt1[1] - 15), (pt1[0] + 75, pt1[1]), color, -1)
            cv2.putText(img, class_name, (pt1[0]+3, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            cv2.putText(img, "{0:.2f}".format(score), (pt1[0]+40, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        else:
            cv2.rectangle(img, (pt1[0], pt1[1] - 15), (pt1[0] + 35, pt1[1]), color, -1)
            cv2.putText(img, class_name[:3], (pt1[0]+3, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

# def visualizeVolume(volume_,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,typ):
#     img_list = []
#     for i in range(0, len(volume_)):
#         img_s = volume_[i].astype(np.uint8)
#         draw_bboxes(img_s,gt,0,LABELMAP)
#         if not (dt is None):
#             dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
#             draw_bboxes(img_s,dt,1,LABELMAP)
#             path_t = os.path.join(path,filename+"_{0}_{1}_result_".format(int(time_stamp_end),i)+typ+".png")
#         else:
#             path_t = os.path.join(path,filename+"_{0}_{1}_".format(int(time_stamp_end),i)+typ+".png")
#         cv2.imwrite(path_t,img_s)
#         # if not(os.path.exists(path_t)):
#         #     os.mkdir(path_t)
#         cv2.imwrite(path_t,img_s)
#         img_list.append(img_s)
#     # img_all = np.stack(img_list).max(0).astype(np.uint8)
#     # if not (dt is None):
#     #     dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
#     #     draw_bboxes(img_all,dt,1,LABELMAP)
#     #     path_t = os.path.join(path,filename+"_{0}_result_all.png".format(int(time_stamp_end)))
#     # else:
#     #     path_t = os.path.join(path,filename+"_{0}_all.png".format(int(time_stamp_end),i))
#     # cv2.imwrite(path_t,img_all)

def visualizeVolume(volume,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    img = np.zeros((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    img[:,:,0] = 119
    img[:,:,2] = 0
    img[:,:,1] = 255
    c_p = volume[5:]
    c_n = volume[:5]
    c = np.stack([c_p,c_n])
    c = np.max(c, 0)
    img_buf = np.zeros_like(c[0])
    
    for i in range(0,5):
        img_0 = (120 + i * 30)+ 119
        tar2 = c[i] / 255 * 255
        tar2 = np.where(tar2 > 255, 255, tar2).astype(np.uint8)
        img[:,:,0] = np.where(c[i] > img_buf, img_0, img[:,:,0])
        img[:,:,2] = np.where(c[i] > img_buf, tar2, img[:,:,2])
        img_buf = np.where(c[i]>img_buf, c[i], img_buf)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    gt = gt[gt['t']==time_stamp_end]
    #draw_bboxes(img,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        #draw_bboxes(img,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_eventvolume_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_eventvolume.png")
    #print(path_t)
    cv2.imwrite(path_t,img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)  #视频流名
    parser.add_argument('-end', type=int)   #标注框时间戳
    parser.add_argument('-suffix', type=str, default="normal")  #一个用于区分参数的后缀
    parser.add_argument('-result_path', type=str, default=None) #summarise.npz路径。不设置时则不包含检测框
    parser.add_argument('-tol', type = int, default=4999)   #检测框和标注框之间的时间容差。以4999时为例，要可视化50000μs位置的标注框时将会同时可视化45001μs到54999μs范围内的检测框
    parser.add_argument('-dataset', type = str, default="gen1") #prophesee gen1/gen4数据集
    parser.add_argument('-data_path', type = str)   #预处理数据（稀疏）到train, val, test这一级的目录
    parser.add_argument('-bbox_path', type = str)   #数据集到train, val, test这一级的目录，用于读取标签
    #可视化结果会输出到"result_allinone/视频流名_标注框时间戳_suffix_datatype.png"（不包含检测框）或"result_allinone/视频流名_标注框时间戳_suffix_datatype_result.png"（包含检测框）

    args = parser.parse_args()

    target_path = 'result_allinone'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    data_folder = 'test'
    item = args.item
    time_stamp_end = args.end

    if args.dataset == "gen1":
        bbox_path = args.bbox_path
        data_path = args.data_path
        result_path = args.result_path
        ori_shape = (240,304)
        shape = (256,320)
        LABELMAP = ["car", "pedestrian"]
    elif args.dataset == "kitti":
        bbox_path = "/home/liubingde/kitti"
        data_path = "/home/liubingde/kitti_taf"
        data_folder = 'val'
        if not (args.exp_name is None):
            result_path = "/home/lbd/100-fps-event-det/" + args.exp_name + "/summarise.npz"
        ori_shape = (375,1242)
        shape = (192,640)
        LABELMAP = ["car", "pedestrian"]
    else:
        bbox_path = args.bbox_path
        data_path = args.data_path
        result_path = args.result_path
        ori_shape = (720,1280)
        shape = (512,640)
        LABELMAP = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']

    if not (args.result_path is None):
        bbox_file = result_path
        f_bbox = np.load(bbox_file)
        dt = f_bbox["dts"][f_bbox["file_names"]==item]
    else:
        dt = None

    final_path = os.path.join(bbox_path,data_folder)
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_"+str(time_stamp_end) + ".npy")
    #print(target)
    locations = np.fromfile(event_file, dtype=np.uint32)
    x = np.bitwise_and(locations, 1023).astype(int)
    y = np.right_shift(np.bitwise_and(locations, 523264), 10).astype(int)
    c = np.right_shift(np.bitwise_and(locations, 3670016), 19).astype(int)
    p = np.right_shift(np.bitwise_and(locations, 4194304), 22).astype(int)
    features = np.right_shift(np.bitwise_and(locations, 2139095040), 23).astype(int)

    events = np.stack([x, y, c, p, features], axis=1)

    C = 5
    volumes = generate_event_volume(events,shape,ori_shape,C)
    #print(np.quantile(volumes[volumes>0],0.05),np.quantile(volumes[volumes>0],0.2),np.quantile(volumes[volumes>0],0.5),np.quantile(volumes[volumes>0],0.75),np.quantile(volumes[volumes>0],0.95))
    visualizeVolume(volumes,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,args.suffix)