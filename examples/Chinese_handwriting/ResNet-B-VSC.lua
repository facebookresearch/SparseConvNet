-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

local sparseconvnet=require 'sparseconvnet'
local tensortype = sparseconvnet.cutorch
and 'torch.CudaTensor' or 'torch.FloatTensor'

-- two-dimensional SparseConvNet
local sparseModel = sparseconvnet.Sequential()
local denseModel = nn.Sequential()
local model = nn.Sequential():add(sparseModel):add(denseModel)

sparseModel
:add(sparseconvnet.ValidConvolution(2,3,16,3,false))
:add(sparseconvnet.MaxPooling(2,2,2))
:add(sparseconvnet.SparseResNet(
    2,16,{
      {'b', 16, 2, 1},
      {'b', 32, 2, 2},
      {'b', 64, 2, 2},
      {'b', 128, 2, 2},}))
sparseModel:add(sparseconvnet.Convolution(2,128,256,4,1,false,false))
sparseModel:add(sparseconvnet.BatchNormReLU(256))
sparseModel:add(sparseconvnet.SparseToDense(2))
denseModel:add(nn.View(256):setNumInputDims(3))
denseModel:add(nn.Linear(256, 3755))
sparseconvnet.initializeDenseModel(denseModel)
model:type(tensortype)
print(model)

inputSpatialSize=sparseModel:suggestInputSize(torch.LongTensor{1,1})
print("inputSpatialSize",inputSpatialSize)

local dataset = dofile('data.lua')(inputSpatialSize,64,3)

sparseconvnet.ClassificationTrainValidate(model,dataset,
  {nEpochs=100,initial_LR=0.1, LR_decay=0.05,weightDecay=1e-4})
