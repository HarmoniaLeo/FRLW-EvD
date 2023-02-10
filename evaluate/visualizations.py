import cv2
import matplotlib as mpl
mpl.use('Agg')  # Required to run the script with "screen" command without a X server
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def draw_bboxes(img, boxes, dt, labelmap):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i,1]), int(boxes[i,2]))
        size = (int(boxes[i,3]), int(boxes[i,4]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes[i,-2]
        class_id = boxes[i,-3]
        class_name = labelmap[int(class_id)]
        color = colors[(dt+1) * 60]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def visualizeVolume(volume,gt,dt,filename,path, classes):
    img = 127 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    c_p = volume[1::2]
    c_p = c_p.sum(axis=0)
    c_n = volume[0::2]
    c_n = c_n.sum(axis=0)
    c_p = np.where(c_p>c_n,c_p,0)
    c_p = c_p/5
    c_p = np.where(c_p>1.0,127.0,c_p*127)
    c_n = np.where(c_n>c_p,c_n,0)
    c_n = c_n/5
    c_n = np.where(c_n>1.0,-127.0,-c_n*127)
    c_map = c_p+c_n
    img_s = img + c_map.astype(np.uint8)[:,:,None]
    draw_bboxes(img_s,gt,0,classes)
    draw_bboxes(img_s,dt,1,classes)
    path_t = os.path.join(path,filename+"_{0}.png".format(int(gt[0,0])))
    cv2.imwrite(path_t,img_s)

def visualize_taf(volume,gt,dt,filename,path, classes):
    ecd = volume[1:volume.shape[0]:2]
    volume = volume[:volume.shape[0]:2]
    ecd_view = ecd[ecd > -1e6]
    q90 = np.quantile(ecd_view, 0.9)
    q10 = np.quantile(ecd_view, 0.1)
    ecd = np.where(ecd > -1e6, ecd - q90, ecd)
    ecd = np.where((ecd > -1e6) & (ecd < 0), ecd/(q90 - q10 + 1e-8) * 2, ecd)
    ecd_view = ecd[ecd > -1e6]
    q100 = np.max(ecd_view) + 1e-8
    ecd = np.where(ecd > 0, ecd / q100 * 2, ecd)
    for i in range(0,volume.shape[0]):
        img_s = 255 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
        tar = ecd[i] + 2.0
        tar = tar / 2.0
        tar = np.where(tar<0,0,tar)
        #tar = np.where(tar * 10 > 1, 1, tar)
        img_0 = (60 * tar).astype(np.uint8) + 119
        #img_1 = (255 * tar).astype(np.uint8)
        #img_2 = (255 * tar).astype(np.uint8)
        img_s[:,:,0] = img_0
        #img_s[:,:,1] = img_1
        #img_s[:,:,2] = img_2
        img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
        draw_bboxes(img_s,gt,0,classes)
        draw_bboxes(img_s,dt,1,classes)
        path_t = os.path.join(path,filename+"_{0}".format(int(gt[0,0])))
        if not(os.path.exists(path_t)):
            os.mkdir(path_t)
        cv2.imwrite(os.path.join(path_t,'{0}.png'.format(i)),img_s)


class visualizer:
    def __init__(self, path, ori_width, ori_height, classes, visualize_method = visualizeVolume):
        self.path = path
        self.visualize_method = visualize_method
        self.ori_width = ori_width
        self.ori_height = ori_height
        self.classes = classes
    
    def visualize(self, volume_t, gt, dt, filename):
        volume_t = volume_t.permute(0,3,1,2).contiguous().view(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])
        volume_t = torch.nn.functional.interpolate(volume_t[None,:],torch.Size((self.ori_height,self.ori_width)))[0]
        volume_t = volume_t.cpu().numpy()
        self.visualize_method(volume_t, gt, dt, filename, self.path, self.classes)