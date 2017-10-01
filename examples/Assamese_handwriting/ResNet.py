# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sparseconvnet as scn
from data import getIterators

# two-dimensional SparseConvNet
class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel=scn.Sequential(
        ).add(scn.ValidConvolution(2, 3, 8, 3, False)
        ).add(scn.MaxPooling(2, 3, 2)
        ).add(scn.SparseResNet(2, 8, [
            ['b', 8, 2, 1],
            ['b', 16, 2, 2],
            ['b', 24, 2, 2],
            ['b', 32, 2, 2]])
        ).add(scn.Convolution(2, 32, 64, 5, 1, False)
        ).add(scn.BatchNormReLU(64)
        ).add(scn.SparseToDense(2,64))
        self.linear = nn.Linear(64, 183)
    def forward(self, x):
        x = self.sparseModel(x)
        x = x.view(-1,64)
        x = self.linear(x)
        return x

model=Model()
spatial_size = model.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
print('Input spatial size:', spatial_size)
dataset = getIterators(spatial_size, 63, 3)
scn.ClassificationTrainValidate(
    model, dataset,
    {'n_epochs': 100,
    'initial_lr': 0.1,
    'lr_decay': 0.05,
    'weight_decay': 1e-4,
    'use_gpu':  torch.cuda.is_available(),
    'check_point': True,})
