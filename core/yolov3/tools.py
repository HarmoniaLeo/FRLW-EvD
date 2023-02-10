import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = 0.5


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        # We ignore those whose tarhets == -1.0. 
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[xc_s, yc_s, anchor_w, anchor_h], ..., [xc_s, yc_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [xc_s, yc_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [xc_s, yc_s, anchor_w, anchor_h] ->  [x1, y1, x2, y2]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # x1
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # y1
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # x2
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # y2
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # x1
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # y1
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # x2
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # y1
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    num_anchors = len(anchor_size)
    anchor_boxes = np.zeros([num_anchors, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def multi_gt_creator(input_size, strides, label_lists, anchor_size):
    """制作训练正样本"""
    batch_size = len(label_lists)
    h = w = input_size
    num_scale = len(strides)
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale

    gt_tensor = []
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    
    # generate gt datas
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            if torch.sum(gt_label) == 0:
                continue
            
            gt_label = gt_label.cpu().numpy()

            gt_class = int(gt_label[0])
            # 计算真实框的中心点和宽高
            # c_x = (xmax + xmin) / 2
            # c_y = (ymax + ymin) / 2
            # box_w = (xmax - xmin)
            # box_h = (ymax - ymin)

            c_x, c_y, box_w, box_h = gt_label[1:]
            xmax = (c_x + box_w / 2) / w
            xmin = (c_x - box_w / 2) / h
            ymax = (c_y + box_h / 2) / w
            ymin = (c_y - box_h / 2) / h

            # 检查数据
            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue    

            # 计算先验框和边界框之间的IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # 阈值筛选
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # 若所有的IoU都小于ignore，则将IoU最大的先验框分配给真实框，其他均视为负样本
                index = np.argmax(iou)
                # 确定该正样本被分配到哪个尺度上去，以及哪个先验框被选中为正样本
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # 获得该尺度的降采样倍数
                s = strides[s_indx]
                # 获得该先验框的参数
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # 计算中心点所在的网格坐标
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # 制作学习标签
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
            else:
                # 至少有一个IoU大于ignore
                
                # 我们只保留IoU最大的作为正样本，
                # 其余的要么被忽略，要么视为负样本
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # 确定该正样本被分配到哪个尺度上去，以及哪个先验框被选中为正样本
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # 获得该尺度的降采样倍数
                            s = strides[s_indx]
                            # 获得该先验框的参数
                            p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            # 计算中心点所在的网格坐标
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            # 制作学习标签
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
                        else:
                            # 这些先验框即便IoU大于ignore，但由于不是最大的
                            # 故被忽略掉
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
        
    return torch.from_numpy(gt_tensor).cuda(non_blocking=True).float()


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)

def giou_score(bboxes_a, bboxes_b, batch_size):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    # iou
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    area_u = area_a + area_b - area_i
    iou = (area_i / (area_u + 1e-14)).clamp(0)
    
    # giou
    tl = torch.min(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.max(bboxes_a[:, 2:], bboxes_b[:, 2:])
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_c = torch.prod(br - tl, 1) * en  # * ((tl < br).all())

    giou = (iou - (area_c - area_u) / (area_c + 1e-14))

    return giou.view(batch_size, -1)
    
def loss(pred_conf, pred_cls, pred_txtytwth, pred_iou, label, num_classes):
    # create loss func
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')
    iou_loss_function = nn.SmoothL1Loss(reduction='none')

    # pred
    pred_conf = pred_conf[:, :, 0]
    #print("pred_conf",torch.max(pred_conf),torch.min(pred_conf))
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    # gt  
    gt_conf = label[:, :, 0]
    #print("gt_conf",torch.unique(gt_conf))
    gt_obj = label[:, :, 1]
    #print("gt_obj", torch.unique(gt_obj))
    gt_cls = label[:, :, 2].long()
    gt_txty = label[:, :, 3:5]
    gt_twth = label[:, :, 5:7]
    gt_box_scale_weight = label[:, :, 7]
    gt_iou = (gt_box_scale_weight > 0.).float()
    gt_mask = (gt_box_scale_weight > 0.).float()

    batch_size = pred_conf.size(0)
    # objectness loss
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    
    # class loss
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size
    
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    bbox_loss = txty_loss + twth_loss

    # iou loss
    iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    #print(conf_loss, cls_loss, bbox_loss)

    return conf_loss, cls_loss, bbox_loss, iou_loss

def label_assignment_with_anchorbox(anchor_size, target_boxes, num_anchors, strides, multi_anchor=False):
    # prepare
    anchor_boxes = set_anchors(anchor_size)
    gt_box = np.array([[0, 0, target_boxes[2], target_boxes[3]]])

    # compute IoU
    iou = compute_iou(anchor_boxes, gt_box)

    label_assignment_results = []
    if multi_anchor:
        # We consider those anchor boxes whose IoU is more than 0.5,
        iou_mask = (iou > 0.5)
        if iou_mask.sum() == 0:
            # We assign the anchor box with highest IoU score.
            iou_ind = np.argmax(iou)

            # scale_ind, anchor_ind = index // num_scale, index % num_scale
            scale_ind = iou_ind // num_anchors
            anchor_ind = iou_ind - scale_ind * num_anchors

            # get the corresponding stride
            stride = strides[scale_ind]

            # compute the grid cell
            xc_s = target_boxes[0] / stride
            yc_s = target_boxes[1] / stride
            grid_x = int(xc_s)
            grid_y = int(yc_s)

            label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])
        else:            
            for iou_ind, iou_m in enumerate(iou_mask):
                if iou_m:
                    # scale_ind, anchor_ind = index // num_scale, index % num_scale
                    scale_ind = iou_ind // num_anchors
                    anchor_ind = iou_ind - scale_ind * num_anchors

                    # get the corresponding stride
                    stride = strides[scale_ind]

                    # compute the gride cell
                    xc_s = target_boxes[0] / stride
                    yc_s = target_boxes[1] / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])

    else:
        # We assign the anchor box with highest IoU score.
        iou_ind = np.argmax(iou)

        # scale_ind, anchor_ind = index // num_scale, index % num_scale
        scale_ind = iou_ind // num_anchors
        anchor_ind = iou_ind - scale_ind * num_anchors

        # get the corresponding stride
        stride = strides[scale_ind]

        # compute the grid cell
        xc_s = target_boxes[0] / stride
        yc_s = target_boxes[1] / stride
        grid_x = int(xc_s)
        grid_y = int(yc_s)

        label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])

    return label_assignment_results


def label_assignment_without_anchorbox(target_boxes, strides):
    # no anchor box
    scale_ind = 0
    anchor_ind = 0

    label_assignment_results = []
    # get the corresponding stride
    stride = strides[scale_ind]

    # compute the grid cell
    xc_s = target_boxes[0] / stride
    yc_s = target_boxes[1] / stride
    grid_x = int(xc_s)
    grid_y = int(yc_s)
    
    label_assignment_results.append([grid_x, grid_y, scale_ind, anchor_ind])
            
    return label_assignment_results


def gt_creator(img_size, strides, label_lists, anchor_size=None, multi_anchor=False, center_sample=False):
    """creator gt"""
    # prepare
    batch_size = len(label_lists)
    img_h = img_w = img_size
    num_scale = len(strides)
    gt_tensor = []
    KA = len(anchor_size) // num_scale if anchor_size is not None else 1

    for s in strides:
        fmp_h, fmp_w = img_h // s, img_w // s
        # [B, H, W, KA, obj+cls+box+scale]
        gt_tensor.append(np.zeros([batch_size, fmp_h, fmp_w, KA, 1+1+4+1]))

    label_lists =  [labels[torch.sum(labels, 1) != 0] for labels in label_lists]
    #print("label_list",label_lists[:5])
    
    # generate gt datas  
    for bi in range(batch_size):
        label = label_lists[bi]
        for box_cls in label:
            box_cls = box_cls.cpu().numpy()
            # get a bbox coords
            #cls_id = int(box_cls[-1])
            cls_id = int(box_cls[0])
            #x1, y1, x2, y2 = box_cls[:-1]
            xc, yc, bw, bh = box_cls[1:]
            # [x1, y1, x2, y2] -> [xc, yc, bw, bh]
            # xc = (x2 + x1) / 2 * img_w
            # yc = (y2 + y1) / 2 * img_h
            # bw = (x2 - x1) * img_w
            # bh = (y2 - y1) * img_h
            x1 = (xc - bw / 2) / img_w
            y1 = (yc - bh / 2) / img_h
            x2 = (xc + bw / 2) / img_w
            y2 = (yc + bh / 2) / img_h
            target_boxes = [xc, yc, bw, bh]
            box_scale = 2.0 - (bw / img_w) * (bh / img_h)

            # check label
            if bw < 1. or bh < 1.:
                # print('A dirty data !!!')
                continue

            # label assignment
            if anchor_size is not None:
                # use anchor box
                label_assignment_results = label_assignment_with_anchorbox(
                                                anchor_size=anchor_size,
                                                target_boxes=target_boxes,
                                                num_anchors=KA,
                                                strides=strides,
                                                multi_anchor=multi_anchor)
            else:
                # no anchor box
                label_assignment_results = label_assignment_without_anchorbox(
                                                target_boxes=target_boxes,
                                                strides=strides)

            # make labels
            for result in label_assignment_results:
                grid_x, grid_y, scale_ind, anchor_ind = result
                
                if center_sample:
                    # We consider four grid points near the center point
                    for j in range(grid_y, grid_y+2):
                        for i in range(grid_x, grid_x+2):
                            if (j >= 0 and j < gt_tensor[scale_ind].shape[1]) and (i >= 0 and i < gt_tensor[scale_ind].shape[2]):
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 0] = 1.0
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 1] = cls_id
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 2:6] = np.array([x1, y1, x2, y2])
                                gt_tensor[scale_ind][bi, j, i, anchor_ind, 6] = box_scale
                else:
                    # We ongly consider top-left grid point near the center point
                    if (grid_y >= 0 and grid_y < gt_tensor[scale_ind].shape[1]) and (grid_x >= 0 and grid_x < gt_tensor[scale_ind].shape[2]):
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 0] = 1.0
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 1] = cls_id
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 2:6] = np.array([x1, y1, x2, y2])
                        gt_tensor[scale_ind][bi, grid_y, grid_x, anchor_ind, 6] = box_scale

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, axis=1)
    
    return torch.from_numpy(gt_tensor).float().cuda()



if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)