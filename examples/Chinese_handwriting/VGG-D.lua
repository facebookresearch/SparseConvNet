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
sparseModel:add(sparseconvnet.SparseVggNetPlus(2,3,{
      {'C', 16}, {'C', 16}, 'MP',
      {'C', 32}, {'C', 32}, 'MP',
      {'C', 64}, {'C', 64}, 'MP',
      {'C', 128}, {'C', 128}, 'MP',
      {'C', 256}, {'C', 256}}))
sparseModel:add(sparseconvnet.Convolution(2,256,512,3,1,false,false))
sparseModel:add(sparseconvnet.BatchNormReLU(512))
sparseModel:add(sparseconvnet.SparseToDense(2))
denseModel:add(nn.View(512):setNumInputDims(3))
denseModel:add(nn.Linear(512, 3755))
sparseconvnet.initializeDenseModel(denseModel)
model:type(tensortype)
print(model)

inputSpatialSize=sparseModel:suggestInputSize(torch.LongTensor{1,1})
print("inputSpatialSize",inputSpatialSize)

local dataset = dofile('data.lua')(inputSpatialSize,63,3)

sparseconvnet.ClassificationTrainValidate(model,dataset,
  {nEpochs=100,initial_LR=0.1, LR_decay=0.05,weightDecay=1e-4})
