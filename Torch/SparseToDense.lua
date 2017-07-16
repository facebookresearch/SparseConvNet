-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Function to convert a SparseConvNet hidden layer to a dense convolutional
layer. Put a SparseToDense convolutional layer (or an ActivePooling layer) at
the top of your sparse network. The output can then pass to a dense
convolutional layers or (if the spatial dimensions have become trivial) a
linear classifier.

Parameters:
dimension : of the input field
]]

return function(sparseconvnet)
  local C = sparseconvnet.C
  local SparseToDense, parent = torch.class(
    'sparseconvnet.SparseToDense', 'nn.Module', sparseconvnet)

  function SparseToDense:__init(dimension)
    parent.__init(self)
    self.dimension = dimension
    self.output=torch.Tensor()
    self.gradInput={features=torch.Tensor()}
  end

  function SparseToDense:updateOutput(input)
    --create a dense nItems x nFeatures x size x ... x size Tensor
    C.dimTypedFn(self.dimension, self._type, 'SparseToDense_updateOutput')(
      input.spatialSize:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.output:cdata(),
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    return self.output
  end

  function SparseToDense:updateGradInput(input, gradOutput)
    C.dimTypedFn(self.dimension, self._type, 'SparseToDense_updateGradInput')(
      input.spatialSize:cdata(),
      input.metadata.ffi,
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput:cdata(),
      self.shared.rulesBuffer and self.shared.rulesBuffer:cdata())
    return self.gradInput
  end

  function SparseToDense:type(type,tensorCache)
    if type==nil then
      return self._type
    end
    return parent.type(self,type,tensorCache)
  end

  function SparseToDense:__tostring()
    return 'SparseToDense'
  end

  function SparseToDense:clearState()
    self.output:set()
    self.gradInput.features:set()
  end

  function SparseToDense:suggestInputSize(nOut)
    return nOut
  end
end
