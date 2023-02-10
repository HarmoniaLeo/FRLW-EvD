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

def generate_event_volume(ecd_file, shape, ori_shape, volume_bins):
    ecds = np.fromfile(ecd_file, dtype=np.uint8).reshape(1, int(volume_bins * 2), shape[0], shape[1]).astype(np.float32)
    ecds = torch.from_numpy(ecds)
    ecds = torch.nn.functional.interpolate(ecds, size = ori_shape, mode='nearest')[0]
    return ecds.numpy()

def generate_taf_gen1(ecd_file, filename, timestamp, shape, ori_shape, volume_bins):
    ecds = []
    for i in range(int(volume_bins)):
        ecd_file_ = os.path.join(os.path.join(ecd_file,"bin{0}".format(7-i)), filename+ "_" + str(timestamp) + ".npy")
        ecd = np.fromfile(ecd_file_, dtype=np.uint8).reshape(2, shape[0], shape[1]).astype(np.float32)
        ecds.append(ecd)

    ecds = torch.from_numpy(np.concatenate(ecds, 0))[None,:,:,:]
    ecds = torch.nn.functional.interpolate(ecds, size = ori_shape, mode='nearest')[0]
    return ecds.numpy()

def generate_taf_gen4(ecd_file, filename, timestamp, shape, ori_shape, volume_bins):
    if volume_bins == 4:
        ecd_file = os.path.join(os.path.join(ecd_file,"bins{0}".format(int(volume_bins))), filename+ "_" + str(timestamp) + ".npy")
        volume = np.fromfile(ecd_file, dtype=np.uint8).reshape(int(volume_bins * 2), shape[0], shape[1]).astype(np.float32)
    else:
        ecd_file1 = os.path.join(os.path.join(ecd_file,"bins{0}".format(int(volume_bins/2))), filename+ "_" + str(timestamp) + ".npy")
        volume = np.fromfile(ecd_file1, dtype=np.uint8).reshape(int(volume_bins), shape[0], shape[1]).astype(np.float32)
        ecd_file2 = os.path.join(os.path.join(ecd_file,"bins{0}".format(int(volume_bins))), filename+ "_" + str(timestamp) + ".npy")
        volume2 = np.fromfile(ecd_file2, dtype=np.uint8).reshape(int(volume_bins), shape[0], shape[1]).astype(np.float32)
        volume = np.concatenate([volume, volume2], 0)

    ecds = torch.from_numpy(volume[None,:,:,:])
    ecds = torch.nn.functional.interpolate(ecds, size = ori_shape, mode='nearest')[0]
    return ecds.numpy()

def generate_optflow(item, time_stamp_end):
    return np.load(os.path.join("optical_flow_buffer",item + "_{0}.npy".format(time_stamp_end)))

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

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    u = np.where(u>np.quantile(u,0.98),np.quantile(u,0.98),u)
    u = np.where(u<np.quantile(u,0.02),np.quantile(u,0.02),u)
    v = np.where(v>np.quantile(v,0.98),np.quantile(v,0.98),v)
    v = np.where(v<np.quantile(u,0.02),np.quantile(v,0.02),v)

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
 
def save_flow(flow, gt,dt,filename,flow_path,time_stamp_end,tol,LABELMAP):
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)
    flow_img = 255 - flow_to_image(flow)

    flow_img = flow_img/255.0  #注意255.0得采用浮点数
    flow_img = (np.power(flow_img,0.4)*255.0).astype(np.uint8)

    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(flow_img,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
        draw_bboxes(flow_img,dt,1,LABELMAP)
        path_t = os.path.join(flow_path,filename+"_{0}".format(int(time_stamp_end)) + "_opticalflow_result.png")
    else:
        path_t = os.path.join(flow_path,filename+"_{0}".format(int(time_stamp_end)) + "_opticalflow.png")
    cv2.imwrite(path_t,flow_img)
 
def extract_flow(flow,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    save_flow(flow, gt,dt,filename,path,time_stamp_end,tol,LABELMAP)

def visualizeTaf(ecds,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    ecd = ecds.max(0)
    img_s = 255 * np.ones((ecds[0].shape[0], ecds[0].shape[1], 3), dtype=np.uint8)
    tar = ecd / 255
    img_0 = (120 * tar).astype(np.uint8) + 119

    tar2 = np.median(ecds, 0) / 180 * 255
    tar2 = np.where(tar2 > 255, 255, tar2).astype(np.uint8)
    #tar2 = np.where(tar2 < 0, 0, tar2).astype(np.uint8)
    #img_1 = (255 * tar).astype(np.uint8)
    #img_2 = (255 * tar).astype(np.uint8)
    img_s[:,:,0] = img_0
    img_s[:,:,2] = tar2
    #img_s[:,:,1] = img_1
    #img_s[:,:,2] = img_2
    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    #mask = np.where(volume[:,:,None] > 1, 1, volume[:,:,None])
    img_s = img_s.astype(np.uint8)
    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(img_s,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_taf_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_taf.png")
    cv2.imwrite(path_t,img_s)

def visualizeE2vid(volume,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    img_s = volume[0]
    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(img_s,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}".format(int(time_stamp_end)) + "_e2vid_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}".format(int(time_stamp_end)) + "_e2vid.png")
    cv2.imwrite(path_t,img_s)

def visualizeFrame(volume,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    img = 127 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    c_p = volume[1]
    c_n = volume[0]
    c_p = c_p/255
    c_n = c_n/255
    c_p = np.where(c_p>c_n,c_p,0)
    c_n = np.where(c_n>c_p,c_n,0)
    c_p = np.where(c_p>1.0,127.0,c_p*127)
    c_n = np.where(c_n>1.0,-127.0,-c_n*127)
    c_map = c_p+c_n
    img_s = img + c_map.astype(np.uint8)[:,:,None]
    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(img_s,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_frame_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_frame.png")
    cv2.imwrite(path_t,img_s)

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
        img_0 = (120 + i * 30).astype(np.uint8) + 119
        tar2 = c[i].astype(np.uint8)
        img[:,:,0] = np.where(c[i] > img_buf, img_0, img[:,:,0])
        img[:,:,2] = np.where(c[i] > img_buf, tar2, img[:,:,2])
        img_buf = np.where(c[i]>img_buf, c[i], img_buf)

    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)

    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(img_s,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_eventvolume_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix + "_eventvolume.png")
    cv2.imwrite(path_t,img_s)

def visualizeTimeSurface(ecds,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,suffix):
    ecd = ecds.max(0)
    #ecd = volume[-1]
    #ecd = volume[-1]
    img_s = 255 * np.ones((ecd.shape[0], ecd.shape[1], 3), dtype=np.uint8)
    #tar = volume[-1] - volume[-2]
    #tar = ecd * 2
    tar = ecd / 255
    #tar = np.where(tar > 1, (tar - 1) / 7 + 1, tar)
    #tar = tar
    #tar = np.where(tar<0,0,tar)
    #tar = np.where(tar * 10 > 1, 1, tar)
    img_0 = (120 * tar).astype(np.uint8) + 119
    #img_1 = (255 * tar).astype(np.uint8)
    #img_2 = (255 * tar).astype(np.uint8)
    img_s[:,:,0] = img_0
    #img_s[:,:,1] = img_1
    #img_s[:,:,2] = img_2
    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    #mask = np.where(volume[:,:,None] > 1, 1, volume[:,:,None])
    img_s = img_s.astype(np.uint8)
    gt = gt[gt['t']==time_stamp_end]
    draw_bboxes(img_s,gt,0,LABELMAP)
    if not (dt is None):
        dt = dt[(dt[:,0]>time_stamp_end-tol)&(dt[:,0]<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix +"_timesurface_result.png")
    else:
        path_t = os.path.join(path,filename+"_{0}_".format(int(time_stamp_end)) + suffix +"_timesurface.png")
    cv2.imwrite(path_t,img_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)  #视频流名
    parser.add_argument('-end', type=int)   #标注框时间戳
    parser.add_argument('-volume_bins', type=float) #Event Representation的通道数/2
    parser.add_argument('-ecd', type=str)   #data_path + ecd = 预处理数据（稠密）到train, val, test这一级的目录
    parser.add_argument('-bbox_path', type=str) #数据集到train, val, test这一级的目录，用于读取标签
    parser.add_argument('-data_path', type=str)   #data_path + ecd = 预处理数据（稠密）到train, val, test这一级的目录
    parser.add_argument('-result_path', type=str, default=None) #summarise.npz路径。不设置时则不包含检测框
    parser.add_argument('-tol', type = int, default=4999)   #检测框和标注框之间的时间容差。以4999时为例，要可视化50000μs位置的标注框时将会同时可视化45001μs到54999μs范围内的检测框
    parser.add_argument('-dataset', type = str, default="gen1") #prophesee gen1/gen4数据集
    parser.add_argument('-datatype', type = str)    #taf/timesurface/frame/eventvolume/opticalflow
    parser.add_argument('-suffix', type = str)  #一个用于区分参数的后缀
    #可视化结果会输出到"result_allinone/视频流名_标注框时间戳_suffix_datatype.png"（不包含检测框）或"result_allinone/视频流名_标注框时间戳_suffix_datatype_result.png"（包含检测框）

    args = parser.parse_args()

    target_path = 'result_allinone'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    data_folder = 'test'
    item = args.item
    time_stamp_end = args.end

    bbox_path = args.bbox_path
    data_path = args.data_path
    result_path = args.result_path
    datatype = args.datatype
    suffix = args.suffix

    if args.dataset == "gen1":
        ori_shape = (240,304)
        shape = (256,320)
        LABELMAP = ["car", "ped"]
    elif args.dataset == "kitti":
        data_folder = 'val'
        ori_shape = (375,1242)
        shape = (192,640)
        LABELMAP = ["car", "ped"]
    else:
        ori_shape = (720,1280)
        shape = (512,640)
        LABELMAP = ['ped', 'cyc', 'car', 'trk', 'bus', 'sign', 'light']

    if not (args.result_path is None):
        bbox_file = result_path
        f_bbox = np.load(bbox_file)
        dt = f_bbox["dts"][f_bbox["file_names"]==item]
    else:
        dt = None

    bbox_path = os.path.join(bbox_path,data_folder)
    bbox_file = os.path.join(bbox_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)


    if datatype == "opticalflow":
        ecds = generate_optflow(item, time_stamp_end)
        extract_flow(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)
    else:
        
        if datatype == "taf":
            if args.dataset == "gen1":
                ecds = generate_taf_gen1(os.path.join(data_path,data_folder), item, time_stamp_end, shape, ori_shape, args.volume_bins)
            elif args.dataset == "gen4":
                ecds = generate_taf_gen4(os.path.join(data_path,data_folder), item, time_stamp_end, shape, ori_shape, args.volume_bins)
            visualizeTaf(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)
        else:
            ecd_file = os.path.join(os.path.join(os.path.join(data_path,args.ecd),data_folder), item+ "_" + str(time_stamp_end) + ".npy")
            ecds = generate_event_volume(ecd_file, shape, ori_shape, args.volume_bins)
            if datatype == "eventvolume":
                visualizeVolume(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)
            elif datatype == "timesurface":
                visualizeTimeSurface(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)
            elif datatype == "e2vid":
                visualizeE2vid(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)
            elif datatype == "frame":
                visualizeFrame(ecds,dat_bbox,dt,item,target_path,time_stamp_end,args.tol,LABELMAP,suffix)