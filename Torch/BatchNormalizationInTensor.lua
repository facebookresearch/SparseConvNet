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
    'sparseconvnet.BatchNormalizationInTensor', 'sparseconvnet.BatchNormalization', sparseconvnet)

  function BN:__init(nPlanes, eps, momentum, outputColumnOffset)
    parent.__init(self,nPlanes,eps,momentum, false, 1)
    self.outputColumnOffset=outputColumnOffset
  end

  function BN:updateOutput(input)
    local o = self.output.features:narrow(2,1+self.outputColumnOffset,self.nPlanes)
    self.output.metadata = input.metadata
    self.output.spatialSize=input.spatialSize
    C.typedFn(self._type,'BatchNormalizationInTensor_updateOutput')(
      input.features:cdata(),
      o:cdata(),
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
    local o = self.output.features:narrow(2,1+self.outputColumnOffset,self.nPlanes)
    local d_o = gradOutput.features:narrow(2,1+self.outputColumnOffset,self.nPlanes)
    C.typedFn(self._type,'BatchNormalization_backward')(
      input.features:cdata(),
      self.gradInput.features:cdata(),
      o:cdata(),
      d_o:cdata(),
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
    local s = 'BatchNormalizationInTensor(' ..
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
end
