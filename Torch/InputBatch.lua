-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C
  local InputBatch, parent = torch.class('sparseconvnet.InputBatch', sparseconvnet)

  function InputBatch:__init(dimension, spatialSize)
    self.dimension = dimension
    self.features = torch.Tensor():type('torch.FloatTensor')
    self.metadata = sparseconvnet.Metadata(dimension)
    self.spatialSize = type(spatialSize)=='number' and torch.LongTensor(
      dimension):fill(spatialSize) or spatialSize
    C.dimensionFn(self.dimension,'setInputSpatialSize')(self.metadata.ffi,
      self.spatialSize:cdata())
  end
  function InputBatch:addSample()
    C.dimensionFn(self.dimension, 'batchAddSample')(self.metadata.ffi)
  end
  function InputBatch:addSampleFromTensor(tensor,offset,threshold)
    C.dimensionFn(
      self.dimension,'addSampleFromThresholdedTensor')(
      self.metadata.ffi, self.features:cdata(), tensor:cdata(), offset:cdata(),
      self.spatialSize:cdata(), threshold)
  end
  function InputBatch:setLocation(location, vector, overwrite)
    --[[location is a self.dimensional length set of coordinates:
    torch.LongStorage or a table]]
    if type(location)=='table' then
      local l=torch.LongStorage(self.dimension)
      for i,x in ipairs(location) do
        l[i]=x
      end
      location = l
    end
    assert(location:min()>=0 and (self.spatialSize-location):min()>0)
    C.dimensionFn(self.dimension,'setInputSpatialLocation')(self.metadata.ffi,
      self.features:cdata(), location:cdata(), vector:cdata(), overwrite)
  end
  function InputBatch:precomputeMetadata(stride)
    if stride==2 then
      C.dimensionFn(self.dimension,'generateRuleBooks2s2')(self.metadata.ffi)
    else
      C.dimensionFn(self.dimension,'generateRuleBooks3s2')(self.metadata.ffi)
    end
  end
  function InputBatch:type(t)
    if t then
      self.features = self.features:type(t)
    else
      return self.features:type()
    end
  end
end
