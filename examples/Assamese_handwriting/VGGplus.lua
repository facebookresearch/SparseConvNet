-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

local sparseconvnet=require 'sparseconvnet'
local tensortype = sparseconvnet.cutorch
and 'torch.CudaTensor' or 'torch.FloatTensor'

-- two-dimensional SparseConvNet
local model = nn.Sequential()
local sparseModel = sparseconvnet.Sequential()
local denseModel = nn.Sequential()
model:add(sparseModel):add(denseModel)
sparseModel:add(sparseconvnet.SparseVggNet(2,3,{
      {'C', 8,}, {'C', 8}, 'MP',
      {'C', 16}, {'C', 16}, 'MP',
      {'C', 16, 8}, {'C', 16, 8}, 'MP',
      {'C', 24, 8}, {'C', 24, 8}, 'MP'}))
sparseModel:add(sparseconvnet.Convolution(2,32,64,5,1,false))
sparseModel:add(sparseconvnet.BatchNormReLU(64))
sparseModel:add(sparseconvnet.SparseToDense(2))
denseModel:add(nn.View(64):setNumInputDims(3))
denseModel:add(nn.Linear(64, 183))
sparseconvnet.initializeDenseModel(denseModel)
model:type(tensortype)
print(model)

inputSpatialSize=sparseModel:suggestInputSize(torch.LongTensor{1,1})
print("inputSpatialSize",inputSpatialSize)

local dataset = dofile('data.lua')(inputSpatialSize,63,3)

sparseconvnet.ClassificationTrainValidate(model,dataset,
  {nEpochs=100,initial_LR=0.1, LR_decay=0.05,weightDecay=1e-4})
