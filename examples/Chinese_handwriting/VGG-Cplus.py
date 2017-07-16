# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.legacy.nn as nn
import sparseconvnet.legacy as scn
from data import getIterators

# Use the GPU if there is one, otherwise CPU
dtype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'

# two-dimensional SparseConvNet
model = nn.Sequential()
sparseModel = scn.Sequential()
denseModel = nn.Sequential()
model.add(sparseModel).add(denseModel)
sparseModel.add(scn.SparseVggNet(2, 3, [
    ['C', 16, 8], ['C', 16, 8], 'MP',
    ['C', 32, 8], ['C', 32, 8], 'MP',
    ['C', 48, 16], ['C', 48, 16], 'MP',
    ['C', 64, 16], ['C', 64, 16], 'MP',
    ['C', 96, 16], ['C', 96, 16]]))
sparseModel.add(scn.Convolution(2, 96 + 16, 128, 3, 2, False))
sparseModel.add(scn.BatchNormReLU(128))
sparseModel.add(scn.SparseToDense(2))
denseModel.add(nn.View(-1, 128))
denseModel.add(nn.Linear(128, 3755))
model.type(dtype)
print(model)

spatial_size = sparseModel.suggestInputSize(torch.LongTensor([1, 1]))
print('input spatial size', spatial_size)
dataset = getIterators(spatial_size, 63, 3)
scn.ClassificationTrainValidate(
    model, dataset,
    {'nEpochs': 100, 'initial_LR': 0.1, 'LR_decay': 0.05, 'weightDecay': 1e-4})
