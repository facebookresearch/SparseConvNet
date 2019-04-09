# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import glob, math, os
import scipy.io
import h5py
import pickle

classes = [
'wall', 'floor', 'cabinet', 'bed',
'chair', 'sofa', 'table', 'door',
'window', 'bookshelf', 'picture', 'counter',
'blinds', 'desk', 'shelves', 'curtain',
'dresser', 'pillow', 'mirror', 'floor mat',
'clothes', 'ceiling', 'books', 'refridgerator',
'television', 'paper', 'towel', 'shower curtain',
'box', 'whiteboard', 'person', 'night stand',
'toilet', 'sink', 'lamp', 'bathtub',
'bag', 'otherstructure', 'otherfurniture', 'otherprop']

corresponding_classes_in_Silberman_labeling = [40, 40,  3, 22,  5, 40, 12, 38, 40, 40,  2, 39, 40, 40, 26, 40, 24,
        40,  7, 40,  1, 40, 40, 34, 38, 29, 40,  8, 40, 40, 40, 40, 38, 40,
        40, 14, 40, 38, 40, 40, 40, 15, 39, 40, 30, 40, 40, 39, 40, 39, 38,
        40, 38, 40, 37, 40, 38, 38,  9, 40, 40, 38, 40, 11, 38, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 13, 40, 40,  6, 40, 23,
        40, 39, 10, 16, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 38, 40, 39, 40, 40, 40, 40, 39, 38, 40, 40, 40, 40, 40, 40, 18,
        40, 40, 19, 28, 33, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 27, 36,
        40, 40, 40, 40, 21, 40, 20, 35, 40, 40, 40, 40, 40, 40, 40, 40, 38,
        40, 40, 40,  4, 32, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 17, 40,
        40, 25, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 38, 38, 40, 40, 39, 40, 39,
        40, 38, 39, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 38,
        40, 40, 38, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        38, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 39, 40, 40, 40, 38, 40, 40, 39, 40, 40, 38, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 31, 40, 40, 40, 40, 40, 40, 40, 38, 40,
        40, 38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 39, 40,
        40, 39, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 38, 39, 40,
        40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        38, 39, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 38,
        40, 40, 40, 38, 40, 39, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        39, 39, 40, 40, 39, 39, 40, 40, 40, 40, 38, 40, 40, 38, 39, 39, 40,
        39, 40, 39, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40,
        38, 40, 39, 40, 40, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 39,
        39, 40, 40, 38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39,
        40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 39, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 38, 40, 40, 40,
        40, 40, 40, 40, 39, 38, 39, 40, 38, 39, 40, 39, 40, 39, 40, 40, 40,
        40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 38, 40, 40, 39, 40, 40,
        40, 39, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 38, 40, 40, 38,
        40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 38, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 38, 38, 40, 40, 40, 38,
        40, 40, 40, 38, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 38, 40, 38, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 38, 38, 40, 40, 40, 38,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40,
        39, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 40,
        40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40,
        40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38,
        40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 38, 40, 39, 40, 40, 40, 40,
        38, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 39, 40, 40]
print(len(classes),len(corresponding_classes_in_Silberman_labeling))
split=scipy.io.loadmat('splits.mat')['testNdxs']-1 # 0-index
testIdxs=[x for x in range(1449) if x in split]
trainIdxs=[x for x in range(1449) if x not in split]
print(len(trainIdxs),len(testIdxs))

f = h5py.File('nyu_depth_v2_labeled.mat','r')
for i,x in enumerate(trainIdxs):
    print(i,x)
    tc=f.get('images')[x]
    td=f.get('depths')[x]*100
    td-=td.mean()
    gt=np.array(f.get('labels')[x],dtype='int16')
    coords=[]
    col=[]
    cl=[]
    for x in range(40,600):
        for y in range(45,470):
            cl.append(corresponding_classes_in_Silberman_labeling[gt[x,y]-1]-1 if gt[x,y]>=1 else -100)
            coords.append([x-320,y-240,td[x,y]])
            col.append([255,tc[0,x,y],tc[1,x,y],tc[2,x,y]])
    coords=np.array(coords,dtype='int16')
    col=np.array(col,dtype='uint8')
    cl=np.array(cl,dtype='int8')
    torch.save([coords,col,cl],'train'+str(i)+'.pth')

f = h5py.File('nyu_depth_v2_labeled.mat','r')
for i,x in enumerate(testIdxs):
    print(i,x)
    tc=f.get('images')[x]
    td=f.get('depths')[x]*100
    td-=td.mean()
    gt=np.array(f.get('labels')[x],dtype='int16')
    coords=[]
    col=[]
    cl=[]
    for x in range(40,600):
        for y in range(45,470):
            cl.append(corresponding_classes_in_Silberman_labeling[gt[x,y]-1]-1 if gt[x,y]>=1 else -100)
            coords.append([x-320,y-240,td[x,y]])
            col.append([255,tc[0,x,y],tc[1,x,y],tc[2,x,y]])
    coords=np.array(coords,dtype='int16')
    col=np.array(col,dtype='uint8')
    cl=np.array(cl,dtype='int8')
    torch.save([coords,col,cl],'test'+str(i)+'.pth')

