from torch import nn
import torch
import torch.nn.functional as F

from core.RED.ssd.anchors.prior_box import PriorBox
from core.RED.ssd.box_head.box_predictor import SSDBoxPredictor
from core.RED.ssd.utils.target_transform import SSDTargetTransform
from core.RED.ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss

class test:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.01
        self.NMS_THRESHOLD = 0.45

class priors:
    def __init__(self, H, W):
        self.STRIDES_x = [32, 64, 128, 213, 320]
        self.STRIDES_y = [32, 64, 128, 256, 512]
        self.FEATURE_MAPS_x = [int(W / stride) for stride in self.STRIDES_x]
        self.FEATURE_MAPS_y = [int(H / stride) for stride in self.STRIDES_y]
        expand = H / 256
        self.MIN_SIZES = [10 * expand, 62 * expand, 114 * expand, 166 * expand, 218 * expand]
        self.MAX_SIZES = [62 * expand, 114 * expand, 166 * expand, 218 * expand, 270 * expand]
        self.ASPECT_RATIOS = [[2, 3], [2, 3], [2, 3], [2], [2]]
        self.BOXES_PER_LOCATION = [6, 6, 6, 4, 4]
        self.CLIP = True

class model:
    def __init__(self, num_classes, H, W):
        self.NEG_POS_RATIO = 3
        self.CENTER_VARIANCE = 0.1
        self.SIZE_VARIANCE = 0.2
        self.THRESHOLD = 0.5
        self.NUM_CLASSES = num_classes
        self.OUT_CHANNELS = [256, 256, 256, 256, 256]
        self.PRIORS = priors(H, W)

class configure:
    def __init__(self, H, W, num_classes):
        self.MODEL = model(num_classes, H, W)
        self.TEST = test()
        self.H = H
        self.W = W

class SSDBoxHead(nn.Module):
    def __init__(self, H, W, num_classes):
        super().__init__()
        cfg = configure(H, W, num_classes + 1)
        self.cfg = cfg
        self.predictor = SSDBoxPredictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO, class_num=num_classes + 1)
        self.post_processor = PostProcessor(cfg)
        self.priors = None
        self.targetTransformer = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)

    def forward(self, features, targets=None, x=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets[:,:,1:], targets[:,:,0]
        gt_boxes_new, gt_labels_new = [], []
        gt_boxes[:,:,0::2] = gt_boxes[:,:,0::2] / self.cfg.W
        gt_boxes[:,:,1::2] = gt_boxes[:,:,1::2] / self.cfg.H
        self.targetTransformer.center_form_priors = self.targetTransformer.center_form_priors.to(targets.device)
        self.targetTransformer.corner_form_priors = self.targetTransformer.corner_form_priors.to(targets.device)
        for gt_boxes_a_batch, gt_labels_a_batch in zip(gt_boxes, gt_labels):
            gt_labels_a_batch = gt_labels_a_batch[gt_boxes_a_batch.sum(1)>0]
            gt_boxes_a_batch = gt_boxes_a_batch[gt_boxes_a_batch.sum(1)>0]
            gt_boxes_a_batch, gt_labels_a_batch = self.targetTransformer(box_utils.center_form_to_corner_form(gt_boxes_a_batch), gt_labels_a_batch + 1)
            gt_boxes_new.append(gt_boxes_a_batch)
            gt_labels_new.append(gt_labels_a_batch)
        gt_boxes, gt_labels = torch.stack(gt_boxes_new), torch.stack(gt_labels_new)
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = (
            reg_loss + cls_loss, 
            reg_loss,
            cls_loss,
        )
        return loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        #boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections
