# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchnet
import sparseconvnet.legacy as scn
import pickle
import math
import random
import numpy
import os

if not os.path.exists('pickle/'):
    print('Downloading and preprocessing data ...')
    os.system(
        'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip')
    os.system(
        'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip')
    os.system('mkdir -p t7/train/ t7/test/ POT/ pickle/')
    os.system('unzip OLHWDB1.1trn_pot.zip -d POT/')
    os.system('unzip OLHWDB1.1tst_pot.zip -d POT/')
    os.system('python3 readPotFiles2.py')


def train(spatial_size, Scale, precomputeStride):
    d = pickle.load(open('pickle/train.pickle', 'rb'))
    d = torchnet.dataset.ListDataset(d)
    randperm = torch.randperm(len(d))

    def perm(idx, size):
        return randperm[idx]

    def merge(tbl):
        inp = scn.InputBatch(2, spatial_size)
        center = spatial_size.float().view(1, 2) / 2
        p = torch.LongTensor(2)
        v = torch.FloatTensor([1, 0, 0])
        for char in tbl['input']:
            inp.addSample()
            for stroke in char:
                stroke = stroke.float() * (Scale - 0.01) / 255 - 0.5 * (Scale - 0.01)
                stroke += center.expand_as(stroke)
                ###############################################################
                # To avoid GIL problems use a helper function:
                scn.dim_fn(
                    2,
                    'drawCurve')(
                    inp.metadata.ffi,
                    inp.features,
                    stroke)
                ###############################################################
                # Above is equivalent to :
                # x1,x2,y1,y2,l=0,stroke[0][0],0,stroke[0][1],0
                # for i in range(1,stroke.size(0)):
                #     x1=x2
                #     y1=y2
                #     x2=stroke[i][0]
                #     y2=stroke[i][1]
                #     l=1e-10+((x2-x1)**2+(y2-y1)**2)**0.5
                #     v[1]=(x2-x1)/l
                #     v[2]=(y2-y1)/l
                #     l=max(x2-x1,y2-y1,x1-x2,y1-y2,0.9)
                #     for j in numpy.arange(0,1,1/l):
                #         p[0]=math.floor(x1*j+x2*(1-j))
                #         p[1]=math.floor(y1*j+y2*(1-j))
                #         inp.setLocation(p,v,False)
                ###############################################################
        inp.precomputeMetadata(precomputeStride)
        return {'input': inp, 'target': torch.LongTensor(tbl['target'])}
    bd = torchnet.dataset.BatchDataset(d, 100, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)

    def iter():
        randperm = torch.randperm(len(d))
        return tdi()
    return iter


def val(spatial_size, Scale, precomputeStride):
    d = pickle.load(open('pickle/test.pickle', 'rb'))
    d = torchnet.dataset.ListDataset(d)
    randperm = torch.randperm(len(d))

    def perm(idx, size):
        return randperm[idx]

    def merge(tbl):
        inp = scn.InputBatch(2, spatial_size)
        center = spatial_size.float().view(1, 2) / 2
        p = torch.LongTensor(2)
        v = torch.FloatTensor([1, 0, 0])
        for char in tbl['input']:
            inp.addSample()
            for stroke in char:
                stroke = stroke.float() * (Scale - 0.01) / 255 - 0.5 * (Scale - 0.01)
                stroke += center.expand_as(stroke)
                scn.dim_fn(
                    2,
                    'drawCurve')(
                    inp.metadata.ffi,
                    inp.features,
                    stroke)
        inp.precomputeMetadata(precomputeStride)
        return {'input': inp, 'target': torch.LongTensor(tbl['target'])}
    bd = torchnet.dataset.BatchDataset(d, 100, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)

    def iter():
        randperm = torch.randperm(len(d))
        return tdi()
    return iter


def getIterators(*args):
    return {'train': train(*args), 'val': val(*args)}
