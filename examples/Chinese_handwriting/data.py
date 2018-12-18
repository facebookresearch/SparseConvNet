
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
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
    os.system(
        'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip')
    os.system(
        'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip')
    os.system('mkdir -p POT/ pickle/')
    os.system('unzip OLHWDB1.1trn_pot.zip -d POT/')
    os.system('unzip OLHWDB1.1tst_pot.zip -d POT/')
    os.system('python readPotFiles.py')

def interp(sample,x,y):
    return torch.from_numpy(np.hstack([np.interp(sample.numpy(),x.numpy(),y[:,i].numpy())[:,None] for i in range(y.shape[1])])).float()
class Data(torch.utils.data.Dataset):
    def __init__(self,file,scale=63):
        print('Loading', file, 'and balancing points for scale', scale)
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
        for i, x in enumerate(self.data):
            x['idx'] = i
        print('Loaded', len(self.data), 'points')
    def __getitem__(self,n):
        return self.data[n]
    def __len__(self):
        return len(self.data)

def MergeFn(spatial_size=63):
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
    return {'train': torch.utils.data.DataLoader(Data('pickle/train.pickle'), collate_fn=MergeFn(), batch_size=100, shuffle=True, num_workers=10),
            'val': torch.utils.data.DataLoader(Data('pickle/test.pickle'), collate_fn=MergeFn(), batch_size=100, shuffle=True, num_workers=10)}
