# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchnet
import sparseconvnet as scn
import pickle
import math
import random
import numpy as np
import os

if not os.path.exists('pickle/'):
    print('Downloading and preprocessing data ...')
    os.system('bash process.sh')
    import process


def train(spatial_size, Scale, precomputeSize):
    d = pickle.load(open('pickle/train.pickle', 'rb'))
    print('Replicating training set 10 times (1 epoch = 10 iterations through the training set = 10x6588 training samples)')
    for i in range(9):
        for j in range(6588):
            d.append(d[j])
    for i, x in enumerate(d):
        x['idx'] = i
    d = torchnet.dataset.ListDataset(d)
    randperm = torch.randperm(len(d))

    def perm(idx, size):
        return randperm[idx]

    def merge(tbl):
        inp = scn.InputBatch(2, spatial_size)
        center = spatial_size.float().view(1, 2) / 2
        p = torch.LongTensor(2)
        v = torch.FloatTensor([1, 0, 0])
        np_random = np.random.RandomState(tbl['idx'])
        for char in tbl['input']:
            inp.add_sample()
            m = torch.eye(2)
            r = np_random.randint(1, 3)
            alpha = random.uniform(-0.2, 0.2)
            if alpha == 1:
                m[0][1] = alpha
            elif alpha == 2:
                m[1][0] = alpha
            else:
                m = torch.mm(m, torch.FloatTensor(
                    [[math.cos(alpha), math.sin(alpha)],
                     [-math.sin(alpha), math.cos(alpha)]]))
            c = center + torch.FloatTensor(1, 2).uniform_(-8, 8)
            for stroke in char:
                stroke = stroke.float() / 255 - 0.5
                stroke = c.expand_as(stroke) + \
                    torch.mm(stroke, m * (Scale - 0.01))
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
                #     for j in np.arange(0,1,1/l):
                #         p[0]=math.floor(x1*j+x2*(1-j))
                #         p[1]=math.floor(y1*j+y2*(1-j))
                #         inp.set_location(p,v,False)
                ###############################################################
        inp.precomputeMetadata(precomputeSize)
        return {'input': inp, 'target': torch.LongTensor(tbl['target']) - 1}
    bd = torchnet.dataset.BatchDataset(d, 108, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)

    def iter():
        randperm.copy_(torch.randperm(len(d)))
        return tdi()
    return iter


def val(spatial_size, Scale, precomputeSize):
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
            inp.add_sample()
            for stroke in char:
                stroke = stroke.float() * (Scale - 0.01) / 255 - 0.5 * (Scale - 0.01)
                stroke += center.expand_as(stroke)
                scn.dim_fn(
                    2,
                    'drawCurve')(
                    inp.metadata.ffi,
                    inp.features,
                    stroke)
        inp.precomputeMetadata(precomputeSize)
        return {'input': inp, 'target': torch.LongTensor(tbl['target']) - 1}
    bd = torchnet.dataset.BatchDataset(d, 183, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)

    def iter():
        randperm.copy_(torch.randperm(len(d)))
        return tdi()
    return iter


def get_iterators(*args):
    return {'train': train(*args), 'val': val(*args)}
