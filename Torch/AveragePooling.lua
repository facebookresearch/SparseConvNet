-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C
  local AveragePooling, parent = torch.class(
    'sparseconvnet.AveragePooling', 'nn.Module', sparseconvnet)

  function AveragePooling:__init(
      dimension, poolSize, poolStride, nFeaturesToDrop)
    parent.__init(self)
    self.dimension = dimension
    self.poolSize = sparseconvnet.toLongTensor(poolSize,dimension)
    self.poolStride = sparseconvnet.toLongTensor(poolStride,dimension)
    self.poolVolume = self.poolSize:prod()
    self.nFeaturesToDrop = nFeaturesToDrop or 0
    self.output = {
      features = torch.FloatTensor(),
    }
    self.gradInput = {
      features = torch.Tensor()
    }
  end

  function AveragePooling:updateOutput(input)
    self.output.metadata=input.metadata
    self.output.spatialSize =
    torch.cdiv(input.spatialSize-self.poolSize,self.poolStride)+1
    C.dimTypedFn(self.dimension, self._type, 'AveragePooling_updateOutput')(
      input.spatialSize:cdata(),
      self.output.spatialSize:cdata(),
      self.poolSize:cdata(),
      self.poolStride:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.output.features:cdata(),
      self.nFeaturesToDrop,
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    return self.output
  end

  function AveragePooling:updateGradInput(input, gradOutput)
    C.dimTypedFn(self.dimension, self._type, 'AveragePooling_updateGradInput')(
      input.spatialSize:cdata(),
      self.output.spatialSize:cdata(),
      self.poolSize:cdata(),
      self.poolStride:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.nFeaturesToDrop,
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    return self.gradInput
  end

  function AveragePooling:type(type,tensorCache)
    if type==nil then
      return self._type
    end
    self._type=type
    self.output.features=self.output.features:type(type)
    self.gradInput.features=self.gradInput.features:type(type)
  end

  function AveragePooling:__tostring()
    local s = 'AveragePooling'
    if self.poolSize:max()==self.poolSize:min()
    and self.poolStride:max()==self.poolStride:min() then
      s=s..self.poolSize[1] ..(self.poolStride[1]==1
        and '' or '/'..self.poolStride[1])
    else
      s=s..'('..self.poolSize[1]
      for i=2,self.dimension do
        s=s..','..self.poolSize[i]
      end
      s=s..')/('..self.poolStride[1]
      for i=2,self.dimension do
        s=s..','..self.poolStride[i]
      end
      s=s..')'
    end
    if self.nFeaturesToDrop>0 then
      s=s .. ' nFeaturesToDrop = ' .. self.nFeaturesToDrop
    end
    return s
  end

  function AveragePooling:clearState()
    self.output={features=self.output.features:set()}
    self.gradInput={features=self.gradInput.features:set()}
  end

  function AveragePooling:suggestInputSize(nOut)
    return torch.cmul(nOut-1,self.poolStride)+self.poolSize
  end
end
