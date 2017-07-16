-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

local sparseconvnet = {}
sparseconvnet.nn=require 'nn'
sparseconvnet.optim=require 'optim'
sparseconvnet.cutorch = pcall(require , 'cutorch')
sparseconvnet.cunn = pcall(require , 'cunn')
sparseconvnet.cudnn = pcall(require, 'cudnn')
for _,module in ipairs({
    'sparseconvnet.C',
    'sparseconvnet.AffineReluTrivialConvolution',
    'sparseconvnet.AveragePooling',
    'sparseconvnet.BatchNormalization',
    'sparseconvnet.BatchNormalizationInTensor',
    'sparseconvnet.BatchwiseDropout',
    'sparseconvnet.CAddTable',
    'sparseconvnet.ClassificationTrainValidate',
    'sparseconvnet.ConcatTable',
    'sparseconvnet.Convolution',
    'sparseconvnet.DataLoader',
    'sparseconvnet.Deconvolution',
    'sparseconvnet.DenseNetBlock',
    'sparseconvnet.Identity',
    'sparseconvnet.InputBatch',
    'sparseconvnet.JoinTable',
    'sparseconvnet.LeakyReLU',
    'sparseconvnet.MaxPooling',
    'sparseconvnet.Metadata',
    'sparseconvnet.NetworkArchitectures',
    'sparseconvnet.NetworkInNetwork',
    'sparseconvnet.ReLU',
    'sparseconvnet.Sequential',
    'sparseconvnet.SparseToDense',
    'sparseconvnet.ValidConvolution',
  }) do
  require(module)(sparseconvnet)
end

function sparseconvnet.initializeDenseModel(model)
  --Following https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nInputPlane --use nInputPlane instead of nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
      else
        v.bias:zero()
      end
    end
  end
  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
    end
  end
  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  BNInit('fbnn.SpatialBatchNormalization')
  BNInit('cudnn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')
  for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end
  return model
end
function sparseconvnet.toLongTensor(x,dimension)
  if type(x) == 'number' then
    return torch.LongTensor(dimension):fill(x)
  elseif type(x) == 'table' then
    return torch.LongTensor(x)
  else
    assert(x:size(1) == dimension)
    return x
  end
end

function sparseconvnet.toDoubleTensor(x,dimension)
  if type(x) == 'number' then
    return torch.DoubleTensor(dimension):fill(x)
  elseif type(x) == 'table' then
    return torch.DoubleTensor(x)
  else
    return x
  end
end

function sparseconvnet.shareShared(mod)
  mod.shared = mod.shared or
  {forwardPassMultiplyAddCount=0, forwardPassHiddenStates=0}
  if mod._type:sub(7,10) == 'Cuda' then --only needed on the GPU
    if sparseconvnet.ruleBookBits==64 then
      mod.shared.rulesBuffer = torch.CudaLongTensor()
    else
      mod.shared.rulesBuffer = torch.CudaIntTensor()
    end
  else
    mod.shared.rulesBuffer = nil
  end
  local function walk(module)
    module.shared=mod.shared
    if module.modules then
      for _,module in ipairs(module.modules) do
        walk(module)
      end
    end
  end
  walk(mod)
end

return sparseconvnet
