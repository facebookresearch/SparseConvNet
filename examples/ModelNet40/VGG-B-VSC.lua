-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

local sparseconvnet=require 'sparseconvnet'
local tensortype = sparseconvnet.cutorch
and 'torch.CudaTensor' or 'torch.FloatTensor'

-- three-dimensional SparseConvNet
local model = nn.Sequential()
local sparseModel = sparseconvnet.Sequential()
local denseModel = nn.Sequential()
model:add(sparseModel):add(denseModel)
sparseModel:add(sparseconvnet.SparseVggNet(3,1,{
      {'C', 16}, {'C', 16}, {'MP',2,2},
      {'C', 32}, {'C', 32}, {'MP',2,2},
      {'C', 64}, {'C', 64}, {'MP',2,2},
      {'C',128}, {'C',128}}))
sparseModel:add(sparseconvnet.Convolution(3,128,128,4+1,1,false,false))
sparseModel:add(sparseconvnet.BatchNormReLU(128))
local nOut=128
sparseModel:add(sparseconvnet.SparseToDense(3))
denseModel:add(nn.View(nOut):setNumInputDims(4))
denseModel:add(nn.Linear(nOut, 40))
sparseconvnet.initializeDenseModel(denseModel)
model:type(tensortype)
print(model)

inputSpatialSize=sparseModel:suggestInputSize(torch.LongTensor{1,1,1})
print("inputSpatialSize",inputSpatialSize)

local dataset = dofile('data.lua')(inputSpatialSize,2)

sparseconvnet.ClassificationTrainValidate(model,dataset,
  {nEpochs=200,initial_LR=0.1, LR_decay=0.025,weightDecay=1e-4})
