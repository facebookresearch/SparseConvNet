# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np


SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

#Predictions will all be in the set {0,1,...,19}
VALID_CLASS_IDS = range(0, len(SELECTED_LABEL_IDS))

#label id to label name mapping: http://dovahkiin.stanford.edu/scannet-public/v1/tasks/scannet-labels.combined.tsv
LABEL_ID_TO_LABEL_NAME = {
    1:	'wall',
    2:	'chair',
    3:	'floor',
    4:	'table',
    5:	'door',
    6:	'couch',
    7:	'cabinet',
    8:	'shelf',
    9:	'desk',
    10:	'office chair',
    11:	'bed',
    12:	'trashcan',
    13:	'pillow',
    14:	'sink',
    15:	'picture',
    16:	'window',
    17:	'toilet',
    18:	'bookshelf',
    19:	'monitor',
    20:	'computer',
    21:	'curtain',
    22:	'book',
    23:	'armchair',
    24:	'coffee table',
    25:	'drawer',
    26:	'box',
    27:	'refrigerator',
    28:	'lamp',
    29:	'kitchen cabinet',
    30:	'dining chair',
    31:	'towel',
    32:	'clothes',
    33:	'tv',
    34:	'nightstand',
    35:	'counter',
    36:	'dresser',
    37:	'countertop',
    38:	'stool',
    39:	'cushion',
}

#Classes relabelled
CLASS_LABELS = []
for i, x in enumerate(SELECTED_LABEL_IDS):
    # print(i, LABEL_ID_TO_LABEL_NAME[x])
    CLASS_LABELS.append(LABEL_ID_TO_LABEL_NAME[x])


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return False
    return (float(tp) / denom, tp, denom)

def evaluate(pred_ids,gt_ids):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids)
    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_iou = get_iou(label_id, confusion)
        if class_iou is not False:
            class_ious[label_name] = get_iou(label_id, confusion)

    sum_iou = 0
    for label_name in class_ious:
        sum_iou+=class_ious[label_name][0]
    mean_iou = sum_iou/len(class_ious)

    print('classes          IoU')
    print('----------------------------')
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        if label_name in class_ious:
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
        else:
            print('{0:<14s}: {1}'.format(label_name, 'missing'))
    print('mean IOU', mean_iou)
