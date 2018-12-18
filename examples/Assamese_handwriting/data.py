# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, torch.utils.data
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


def interp(sample,x,y):
    return torch.from_numpy(np.hstack([np.interp(sample.numpy(),x.numpy(),y[:,i].numpy())[:,None] for i in range(y.shape[1])])).float()
class Data(torch.utils.data.Dataset):
    def __init__(self,file,scale=63,repeats=1):
        torch.utils.data.Dataset.__init__(self)
        self.data = pickle.load(open(file, 'rb'))
        for j in range(len(self.data)):
            strokes=[]
            features=[]
            for k,stroke in enumerate(self.data[j]['input']):
                if len(stroke)>1:
                    stroke=stroke.float()/255-0.5
                    stroke*=scale-1e-3
                    delta=stroke[1:]-stroke[:-1]
                    mag=(delta**2).sum(1)**0.5
                    l=mag.cumsum(0)
                    zl=torch.cat([torch.zeros(1),l])
                    strokes.append(interp(torch.arange(0,zl[-1]),zl,stroke))
                    delta/=mag[:,None]
                    delta=torch.Tensor(delta[[i//2 for i in range(2*len(l))]])
                    zl_=zl[[i//2 for i in range(1,2*len(l)+1)]]
                    features.append(interp(torch.arange(0,zl[-1]),zl_,delta))
            self.data[j]['coords'] = torch.cat(strokes,0)
            self.data[j]['features'] = torch.cat(features,0)
            self.data[j]['target']-=1
        if repeats>1:
            print('Replicating dataset: 1 epoch = %d iterations of the dataset; %d x %d = %d training samples'%(repeats, repeats, len(self.data), repeats * len(self.data)))
        for j in range(len(self.data)):
            for i in range(repeats-1):
                self.data.append(self.data[j])
        for i, x in enumerate(self.data):
            x['idx'] = i
    def __getitem__(self,n):
        return self.data[n]
    def __len__(self):
        return len(self.data)
def TrainMergeFn(spatial_size=95, jitter=8):
    center = spatial_size/2
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            m = torch.eye(2)
            r = torch.randint(0,3,[1]).int().item()
            alpha = torch.rand(1).item()*0.4-0.2
            if r == 1:
                m[0][1] = alpha
            elif r == 2:
                m[1][0] = alpha
            else:
                m = torch.mm(m, torch.FloatTensor(
                    [[math.cos(alpha), math.sin(alpha)],
                     [-math.sin(alpha), math.cos(alpha)]]))
            coords=char['coords']
            coords = torch.mm(coords, m) + torch.FloatTensor(1, 2).uniform_(center-jitter, center+jitter)
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            f=char['features']
            f=torch.mm(f, m)
            f /= (f**2).sum(1,keepdim=True)**0.5
            f = torch.cat([f,torch.ones([f.size(0),1])],1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge
def TestMergeFn(spatial_size=95):
    center = spatial_size/2
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            coords=char['coords']+center
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            f=char['features']
            f = torch.cat([f,torch.ones([f.size(0),1])],1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge


def get_iterators(*args):
    return {'train': torch.utils.data.DataLoader(Data('pickle/train.pickle',repeats=10), collate_fn=TrainMergeFn(), batch_size=108, shuffle=True, num_workers=10),
            'val': torch.utils.data.DataLoader(Data('pickle/test.pickle',repeats=1), collate_fn=TestMergeFn(), batch_size=183, shuffle=True, num_workers=10)}
