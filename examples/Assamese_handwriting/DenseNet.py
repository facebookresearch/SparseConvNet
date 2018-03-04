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
sparseModel.add(scn.SubmanifoldConvolution(2, 3, 8, 3, False))\
    .add(scn.SparseDenseNet(2, 8, [
        {'pool': 'MP', 'compression': 0},
        {'nExtraLayers': 2, 'growthRate': 8},
        {'pool': 'BN-R-C-AP', 'compression': 0},
        {'nExtraLayers': 2, 'growthRate': 8},
        {'pool': 'BN-R-C-AP', 'compression': 0},
        {'nExtraLayers': 2, 'growthRate': 8},
        {'pool': 'BN-R-C-AP', 'compression': 0},
        {'nExtraLayers': 2, 'growthRate': 8}]))
n_out = sparseModel.modules[-1].nOutputPlanes
sparseModel.add(scn.Convolution(2, n_out, 64, 6, 1, False))
sparseModel.add(scn.BatchNormReLU(64))
sparseModel.add(scn.SparseToDense(2))
denseModel.add(nn.View(-1, 64))
denseModel.add(nn.Linear(64, 183))
model.type(dtype)
print(model)

spatial_size = sparseModel.suggestInputSize(torch.LongTensor([1, 1]))
print('input spatial size', spatial_size)
dataset = getIterators(spatial_size, 63, 2)
scn.ClassificationTrainValidate(
    model, dataset,
    {'nEpochs': 100, 'initial_LR': 0.1, 'LR_decay': 0.05, 'weightDecay': 1e-4})
