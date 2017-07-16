-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C

  local DenseNetBlock, parent = torch.class(
    'sparseconvnet.DenseNetBlock', 'nn.Container', sparseconvnet)

  function DenseNetBlock:__init(dimension, nInputPlanes, nExtraLayers,
      growthRate)
    parent.__init(self)
    self.dimension=dimensions
    self.nInputPlanes=nInputPlanes
    self.nExtraLayers=nExtraLayers or 2
    self.growthRate=growthRate or 16
    assert(self.nExtraLayers>=1)
    self.nOutputPlanes=nInputPlanes+nExtraLayers*growthRate

    self.output={
      features=torch.Tensor(), --nActive x self.nOutputPlanes
    }

    --Module 1: Batchnorm the input into the start of self.output
    self:add(sparseconvnet.BatchNormalizationInTensor(nInputPlanes,nil,nil,0))
    self.modules[1].output=self.output
    self.gradInput=self.modules[1].gradInput

    for i = 1, nExtraLayers do
      local nFeatures = self.nInputPlanes + (i-1)*growthRate
      local nFeaturesB=4*growthRate
      --Modules 4*i-2
      self:add(sparseconvnet.AffineReluTrivialConvolution(nFeatures, nFeaturesB, true))
      --Module 4*i-1
      self:add(sparseconvnet.BatchNormalization(nFeaturesB,nil,nil,true,0))
      --Module 4*i
      self:add(sparseconvnet.ValidConvolution(dimension, nFeaturesB, growthRate,
          3, false))
      --Module 4*i+1
      self:add(sparseconvnet.BatchNormalizationInTensor(growthRate,nil,nil,
          self.nInputPlanes+(i-1)*growthRate))
      self.modules[4*i+1].output=self.output
    end

    self.filterSize = self.modules[4].filterSize
    self.filterStride = self.modules[4].filterStride
    self.filterSizeString = self.modules[4].filterSizeString
  end

  function DenseNetBlock:updateOutput(input)
    assert(input.features:size(2) == self.nInputPlanes)
    self.output.spatialSize = input.spatialSize
    self.output.metadata = input.metadata
    self.output.features:resize(input.features:size(1),self.nOutputPlanes)
    local i = input
    for m = 1, 4*self.nExtraLayers+1 do
      i=self.modules[m]:updateOutput(i)
    end
    return self.output
  end

  function DenseNetBlock:backward(input, gradOutput)
    local g = gradOutput
    for i = 1, self.nExtraLayers do
      self.modules[4*i-2].gradInput=gradOutput
    end
    for m=4*self.nExtraLayers+1,2,-1 do
      g = self.modules[m]:backward(self.modules[m-1].output,g)
    end
    self.modules[1]:backward(input,g)
    return self.gradInput
  end

  function DenseNetBlock:type(type,tensorCache)
    self._type=type
    self.output.features=self.output.features:type(type)
    for _,x in pairs(self.modules) do
      x:type(type)
    end
  end

  function DenseNetBlock:__tostring()
    local s = 'DenseNetBlock('.. self.nInputPlanes .. '->' ..
    self.nInputPlanes .. '+' .. self.nExtraLayers .. '*' ..
    self.growthRate .. '=' .. self.nOutputPlanes .. ')'
    return s
  end

  function DenseNetBlock:clearState()
    for _,m in ipairs(self.modules) do
      m:clearState()
    end
    self.output={
      features=self.output.features:set(),
      nPlanes=self.nOutputPlanes,
      dimension=self.dimension
    }
  end

  function DenseNetBlock:suggestInputSize(nOut)
    return nOut
  end
end
