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
      {'C', 16,}, {'C', 16}, 'MP',
      {'C', 32}, {'C', 32}, 'MP',
      {'C', 48}, {'C', 48}, 'MP',
      {'C', 64}, {'C', 64}, 'MP',
      {'C', 96}, {'C', 96}}))
sparseModel:add(sparseconvnet.Convolution(2,96,128,3,2,false))
sparseModel:add(sparseconvnet.BatchNormReLU(128))
sparseModel:add(sparseconvnet.SparseToDense(2))
denseModel:add(nn.View(128):setNumInputDims(3))
denseModel:add(nn.Linear(128, 3755))
sparseconvnet.initializeDenseModel(denseModel)
model:type(tensortype)
print(model)

inputSpatialSize=sparseModel:suggestInputSize(torch.LongTensor{1,1})
print("inputSpatialSize",inputSpatialSize)

local dataset = dofile('data.lua')(inputSpatialSize,63,3)

sparseconvnet.ClassificationTrainValidate(model,dataset,
  {nEpochs=100,initial_LR=0.1, LR_decay=0.05,weightDecay=1e-4})
