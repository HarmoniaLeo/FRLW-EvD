import numpy as np

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def single_infer_gl(labels, timestamps, tol):
    max_labels = 20
    padded_labels = np.zeros((len(timestamps),max_labels, labels.shape[1]),dtype=float)
    for i,timestamp in enumerate(timestamps):
        labels_ = labels[(labels[:,5] + tol >= timestamp)&(labels[:,5] - tol <= timestamp)]
        assert max_labels >= len(labels_)
        padded_labels[i, range(len(labels_))] = labels_.astype(float)
    return padded_labels

def multi_infer_gl(labels, timestamps, tol):
    max_labels = 20
    padded_labels = np.zeros((len(timestamps),max_labels, labels.shape[1]),dtype=float)
    for i,timestamp in enumerate(timestamps):
        distance = np.abs(labels[:,5] - timestamp)
        min_ind = distance.argmin()
        matched_timestamp = labels_[min_ind, 5]
        labels_ = labels_[labels[:,5] == matched_timestamp]
        assert max_labels >= len(labels_)
        padded_labels[i, range(len(labels_))] = labels_.astype(float)
    return padded_labels