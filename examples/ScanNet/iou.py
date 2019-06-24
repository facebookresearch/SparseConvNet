# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np

#VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

#Classes relabelled {-100,0,1,...,19}.
#Predictions will all be in the set {0,1,...,19}


CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
UNKNOWN_ID = -100
N_CLASSES = len(CLASS_LABELS)

def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def evaluate(pred_ids,gt_ids):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids)
    class_ious = {}
    mean_iou = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        mean_iou+=class_ious[label_name][0]/20

    print('classes          IoU')
    print('----------------------------')
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
    print('mean IOU', mean_iou)
