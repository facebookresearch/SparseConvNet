-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Parameters:
nPlanes : number of input planes
eps : small number used to stabilise standard deviation calculation
momentum : for calculating running average for testing (default 0.9)
affine : only 'true' is supported at present (default 'true')
noise : add multiplicative and additive noise during training if >0.
leakiness : Apply activation function inplace: 0<=leakiness<=1.
0 for ReLU, values in (0,1) for LeakyReLU, 1 for no activation function.
]]
return function(sparseconvnet)
  local C = sparseconvnet.C
  local BN,parent = torch.class(
    'sparseconvnet.BatchNormalization', 'nn.Module', sparseconvnet)

  function BN:__init(nPlanes, eps, momentum, affine, leakiness)
    parent.__init(self)
    assert(nPlanes%4==0)
    self.nPlanes=nPlanes
    self.leakiness=leakiness or 1
    if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
    else
      self.affine = true
    end
    self.eps = eps or 1e-5
    self.saveMean = torch.Tensor(nPlanes)
    self.saveInvStd = torch.Tensor(nPlanes)
    self.momentum = momentum or 0.9
    self.runningMean = torch.Tensor(nPlanes)
    self.runningVar = torch.Tensor(nPlanes)
    if self.affine then
      self.weight = torch.Tensor(nPlanes)
      self.bias = torch.Tensor(nPlanes)
      self.gradWeight = torch.Tensor(nPlanes)
      self.gradBias = torch.Tensor(nPlanes)
    end
    self.output = {
      features = torch.Tensor()
    }
    self.gradInput = {
      features = torch.Tensor()
    }
    self:reset()
  end
  function BN:reset()
    if self.affine then
      self.weight:fill(1)
      self.bias:zero()
    end
    self.runningMean:zero()
    self.runningVar:fill(1)
    self.saveMean:zero()
    self.saveInvStd:fill(1)
  end

  function BN:updateOutput(input)
    assert(input.features:size(2)==self.nPlanes)
    self.output.metadata = input.metadata
    self.output.spatialSize=input.spatialSize
    C.typedFn(self._type,'BatchNormalization_updateOutput')(
      input.features:cdata(),
      self.output.features:cdata(),
      self.saveMean:cdata(),
      self.saveInvStd:cdata(),
      self.runningMean:cdata(),
      self.runningVar:cdata(),
      self.weight and self.weight:cdata(),
      self.bias and self.bias:cdata(),
      self.eps,
      self.momentum,
      self.train,
      self.leakiness)
    return self.output
  end

  function BN:backward(input, gradOutput)
    assert(self.train)
    C.typedFn(self._type,'BatchNormalization_backward')(
      input.features:cdata(),
      self.gradInput.features:cdata(),
      self.output.features:cdata(),
      gradOutput.features:cdata(),
      self.saveMean:cdata(),
      self.saveInvStd:cdata(),
      self.runningMean:cdata(),
      self.runningVar:cdata(),
      self.weight and self.weight:cdata(),
      self.bias and self.bias:cdata(),
      self.gradWeight and self.gradWeight:cdata(),
      self.gradBias and self.gradBias:cdata(),
      self.leakiness)
    return self.gradInput
  end

  function BN:updateGradInput(input, gradOutput)
    assert(false) --just call backward
  end

  function BN:accGradParameters(input, gradOutput, scale)
    assert(false) --just call backward
  end

  function BN:__tostring()
    local l
    if self.leakiness==0 then
      l=',ReLU'
    elseif self.leakiness==1/3 then
      l=',LeakyReLU(0.333..)'
    elseif self.leakiness<1 then
      l=',LeakyReLU('..self.leakiness..')'
    else
      l=''
    end
    local s = 'BatchNormalization(' ..
    'nPlanes=' .. self.nPlanes..',' ..
    'eps=' .. self.eps .. ',' ..
    'momentum=' .. self.momentum .. l .. ')'
    return s
  end

  function BN:clearState()
    self.output={features=self.output.features:set()}
    self.gradInput={features=self.gradInput.features:set()}
  end

  function BN:suggestInputSize(nOut)
    return nOut
  end

  local BN,parent = torch.class('sparseconvnet.BatchNormReLU',
    'sparseconvnet.BatchNormalization', sparseconvnet)
  function BN:__init(nPlanes, eps, momentum)
    parent.__init(self, nPlanes, eps, momentum, true, 0)
  end
end
