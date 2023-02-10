import torch
import torchvision
import numpy as np
from evaluate.src.io.box_filtering import filter_boxes_gen1, filter_boxes_large, filter_boxes_kitti
from evaluate.src.metrics.coco_eval import evaluate_detection
import os
import pandas as pd

class evaluator:
    def __init__(self, classes, batchsize, infer_time, ori_width, ori_height, input_width, input_height, dataset = "gen1", recorder = None):
        self.dt_to_eval = []
        self.gt_to_eval = []
        # self.dt = [[] for i in range(batchsize)]
        # self.gt = [[] for i in range(batchsize)]
        self.rw = ori_width / input_width
        self.rh = ori_height / input_height
        self.ori_width = ori_width
        self.ori_height = ori_height
        self.batchsize = batchsize
        self.infer_time = 0
        self.represent_time = 0
        self.infer_count = 0
        self.first_batch = True
        self.classes = classes
        if dataset == "gen1":
            self.filter_boxes = filter_boxes_gen1
        elif dataset == "kitti":
            self.filter_boxes = filter_boxes_kitti
        else:
            self.filter_boxes = filter_boxes_large
        self.tol = int(infer_time/2 - 1)
        self.recorder = recorder
    
    def cal_time(self, infer_time, represent_time):
        if self.first_batch:
            self.first_batch = False
        else:
            self.infer_time += infer_time
            self.represent_time += represent_time
            self.infer_count += 1
            #print(self.infer_time/self.infer_count)

    def transform_gt(self, bounding_box):
        gt_trans=bounding_box.cpu().numpy()
        gt_trans=gt_trans[(gt_trans[:,6]>0)]
        gt_trans=np.array([gt_trans[:,5],
                            (gt_trans[:,0]-gt_trans[:,2]/2)*self.rw,
                            (gt_trans[:,1]-gt_trans[:,3]/2)*self.rh,
                            gt_trans[:,2]*self.rw,
                            gt_trans[:,3]*self.rh,
                            gt_trans[:,4],
                            gt_trans[:,6],
                            gt_trans[:,7]]).T
        return gt_trans
    
    def transform_dt(self, detected_bbox, bins_time_stamp):
        dt_trans = torch.cat([(detected_bbox[...,0:1]-detected_bbox[...,2:3]/2)*self.rw,
                                (detected_bbox[...,1:2]-detected_bbox[...,3:4]/2)*self.rh,
                                detected_bbox[...,2:3]*self.rw,
                                detected_bbox[...,3:4]*self.rh,
                                detected_bbox[...,4:]],dim=-1).cpu().numpy()
        dt_trans=np.concatenate([np.zeros_like(dt_trans[:,:1]) + bins_time_stamp,dt_trans,np.zeros_like(dt_trans[:,:1])],axis=1)
        return dt_trans
    
    def add_result(self, outputs, bins_time_stamps, bounding_box, filename, infer_time, represent_time):
        self.cal_time(infer_time, represent_time)
        for i in range(len(outputs)):
            gt_trans = self.transform_gt(bounding_box[i])
            if len(gt_trans) == 0:
                continue
            #self.gt[i].append(gt_trans)
            self.gt_to_eval.append(gt_trans)
            dt_trans = self.transform_dt(outputs[i], bins_time_stamps[i])
            #self.dt[i].append(dt_trans)
            self.dt_to_eval.append(dt_trans)
            # print(gt_trans, dt_trans)
            # eval_results = evaluate_detection([gt_trans], [dt_trans], time_tol = self.tol, classes=self.classes,height=self.ori_height,width=self.ori_width)
            if not (self.recorder is None):
                self.recorder.record(dt_trans, filename[i])
    
    def end_a_batch(self):
        # dt = [np.concatenate(d) for d in self.dt if len(d)>0]
        # gt = [np.concatenate(g) for g in self.gt if len(g)>0]
        # self.dt_to_eval = self.dt_to_eval + dt
        # self.gt_to_eval = self.gt_to_eval + gt
        # self.dt = [[] for i in range(self.batchsize)]
        # self.gt = [[] for i in range(self.batchsize)]
        pass
    
    def evaluate(self):
        gt_boxes_list = map(self.filter_boxes, self.gt_to_eval)
        result_boxes_list = map(self.filter_boxes, self.dt_to_eval)
        gt_boxes_list1 = []
        result_boxes_list1 = []
        for l1,l2 in zip(gt_boxes_list,result_boxes_list):
            if len(l1) > 0:
                gt_boxes_list1.append(l1)
                if len(l2) == 0:
                    result_boxes_list1.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
                else:
                    result_boxes_list1.append(l2)
        
        eval_results = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = self.tol, classes=self.classes,height=self.ori_height,width=self.ori_width)

        a_infer_time = 1000 * self.infer_time / self.infer_count
        a_represent_time = 1000 * self.represent_time / self.infer_count

        print("Average infer time: {:.2f} ms. ".format(a_infer_time))
        print("Average represent time: {:.2f} ms. ".format(a_represent_time))

        if not (self.recorder is None):
            self.recorder.save()
        
        print("Current score: ",eval_results[0])

        return eval_results

class recorder:
    def __init__(self, save_path):
        self.data_names = []
        self.dt = []
        self.save_path = save_path
        # self.res = 0
        # self.iter_count = 0

    def record(self, dt_trans, file_name):
        # self.iter_count += 1
        # self.res += eval_results[0]
        #print(self.res/self.iter_count)
        for j in range(len(dt_trans)):
            self.data_names.append(file_name)
            self.dt.append(dt_trans[j])
    
    def save(self):
        csv_path = os.path.join(self.save_path,"summarise.npz")
        np.savez(csv_path,
            file_names = self.data_names,
            dts = self.dt)
        print("Summarise to: " + csv_path)