-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C

  local Convolution, parent = torch.class(
    'sparseconvnet.ValidConvolution', 'nn.Module', sparseconvnet)

  function Convolution:__init(dimension, nInputPlanes, nOutputPlanes,
      filterSize, bias)
    parent.__init(self)
    self.dimension = dimension
    self.nInputPlanes = nInputPlanes
    self.nOutputPlanes = nOutputPlanes
    self.filterSize = sparseconvnet.toLongTensor(filterSize,dimension)
    self.filterStride = sparseconvnet.toLongTensor(1,dimension)
    self.filterVolume = self.filterSize:prod()
    for i = 1, dimension do
      assert(self.filterSize[i]%2==1)
    end
    self.weight = torch.Tensor(nInputPlanes*self.filterVolume,nOutputPlanes)
    self.gradWeight = torch.Tensor(nInputPlanes*self.filterVolume,nOutputPlanes)
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

  function Convolution:reset()
    local stdv = math.sqrt(2/self.nInputPlanes/self.filterVolume)
    self.weight:normal(0, stdv)
    if self.bias then
      self.bias:zero()
    end
    return self
  end

  function Convolution:updateOutput(input)
    assert(input.features:size(2)==self.nInputPlanes)
    self.output.metadata = input.metadata
    self.output.spatialSize = input.spatialSize
    self.shared.forwardPassMultiplyAddCount=
    self.shared.forwardPassMultiplyAddCount+
    C.dimTypedFn(self.dimension, self._type, 'ValidConvolution_updateOutput')(
      input.spatialSize:cdata(),
      self.filterSize:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.output.features:cdata(),
      self.weight:cdata(),
      self.bias and self.bias:cdata(),
      self.filterVolume,
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    self.shared.forwardPassHiddenStates=
    self.shared.forwardPassHiddenStates + self.output.features:nElement()
    return self.output
  end

  function Convolution:backward(input, gradOutput)
    C.dimTypedFn(self.dimension, self._type, 'ValidConvolution_backward')(
      input.spatialSize:cdata(),
      self.filterSize:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.weight:cdata(),
      self.gradWeight:cdata(),
      self.gradBias and self.gradBias:cdata() ,
      self.filterVolume,
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    return self.gradInput
  end

  function Convolution:type(type,tensorCache)
    if type==nil then
      return self._type
    else
      self._type=type
      self.weight = self.weight:type(type)
      self.gradWeight =self.gradWeight:type(type)
      if self.bias then
        self.bias = self.bias:type(type)
        self.gradBias =self.gradBias:type(type)
      end
      self.output.features=self.output.features:type(type)
      self.gradInput.features=self.gradInput.features:type(type)
    end
  end

  function Convolution:__tostring()
    local s = 'ValidConvolution ' .. self.nInputPlanes .. '->' .. self.nOutputPlanes..' C'
    if self.filterSize:max()==self.filterSize:min() and
    self.filterStride:max()==self.filterStride:min() then
      s=s..self.filterSize[1] ..(self.filterStride[1]==1 and
        '' or '/'..self.filterStride[1])
    else
      s=s..'('..self.filterSize[1]
      for i=2,self.dimension do
        s=s..','..self.filterSize[i]
      end
      s=s..')/('..self.filterStride[1]
      for i=2,self.dimension do
        s=s..','..self.filterStride[i]
      end
      s=s..')'
    end
    return s
  end

  function Convolution:clearState()
    self.output={features=self.output.features:set()}
    self.gradInput={features=self.gradInput.features:set()}
    self.rules=nil
  end

  function Convolution:suggestInputSize(nOut)
    return nOut
  end
end
