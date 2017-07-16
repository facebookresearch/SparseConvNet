-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C
  local math = require 'math'
  local NetworkInNetwork, parent = torch.class(
    'sparseconvnet.NetworkInNetwork', 'nn.Module', sparseconvnet)

  function NetworkInNetwork:__init(nInputPlanes,nOutputPlanes,bias)
    parent.__init(self)
    self.nInputPlanes = nInputPlanes
    self.nOutputPlanes = nOutputPlanes
    self.weight = torch.Tensor(nInputPlanes,nOutputPlanes)
    self.gradWeight = torch.Tensor(nInputPlanes,nOutputPlanes)
    if (type(bias) ~= 'boolean') or bias then
      self.bias = torch.Tensor(nOutputPlanes)
      self.gradBias = torch.Tensor(nOutputPlanes)
    end
    self.output = {
      features = torch.Tensor(),
    }
    self.gradInput = {
      features = torch.Tensor()
    }
    self:reset()
  end

  function NetworkInNetwork:reset()
    local stdv = math.sqrt(2/self.nInputPlanes)
    if self._type=='torch.CudaDoubleTensor' then
      self.weight = torch.CudaTensor(
        self.weight:size()):normal(0,stdv):type(self._type)
    else
      self.weight:normal(0, stdv)
    end
    if self.bias then
      self.bias:zero()
    end
    return self
  end

  function NetworkInNetwork:updateOutput(input)
    self.output.metadata = input.metadata
    self.output.spatialSize = input.spatialSize
    self.shared.forwardPassMultiplyAddCount=
    self.shared.forwardPassMultiplyAddCount+
    C.typedFn(self._type,'NetworkInNetwork_updateOutput')(
      input.features:cdata(),
      self.output.features:cdata(),
      self.weight:cdata(),
      self.bias and self.bias:cdata())
    self.shared.forwardPassHiddenStates =
    self.shared.forwardPassHiddenStates + self.output.features:nElement()
    return self.output
  end

  function NetworkInNetwork:updateGradInput(input, gradOutput)
    C.typedFn(self._type, 'NetworkInNetwork_updateGradInput')(
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.weight:cdata())

    return self.gradInput
  end

  function NetworkInNetwork:accGradParameters(input, gradOutput, scale)
    assert(not scale or scale==1)
    C.typedFn(self._type, 'NetworkInNetwork_accGradParameters')(
      input.features:cdata(),
      gradOutput.features:cdata(),
      self.gradWeight:cdata(),
      self.gradBias and self.gradBias:cdata())
  end

  function NetworkInNetwork:__tostring()
    local s = 'NetworkInNetwork('..self.nInputPlanes..','..
    self.nOutputPlanes..')'
    return s
  end

  function NetworkInNetwork:clearState()
    self.output = {
      features = self.output.features:set(),
    }
    self.gradInput = {
      features = self.gradInput.features:set()
    }
  end

  function NetworkInNetwork:suggestInputSize(nOut)
    return nOut
  end
end
