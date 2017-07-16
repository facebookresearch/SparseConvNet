-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Assume all the inputs have identical SparseGrids and input[i].nActive
Assume input[1].nPlanes >= input[i].nPlanes for all i=1,#input
output.validRules is taken from input[1].validRules (could do set union?)
(for resnets, make sure the residual link is input[2])
]]

return function(sparseconvnet)
  local CAddTable, parent = torch.class(
    'sparseconvnet.CAddTable', 'nn.Module', sparseconvnet)

  function CAddTable:__init(ip)
    parent.__init(self)
    self.inplace = type(ip)=='boolean' and ip
    self.gradInput = {}
    self.output = self.inplace and 'recycle' or {
      features = torch.Tensor()
    }
    sparseconvnet.shareShared(self)
  end

  function CAddTable:add(module)
    table.insert(self.modules,module)
    sparseconvnet.shareShared(self)
    return self
  end

  function CAddTable:updateOutput(input)
    if self.inplace then
      self.output=input[1]
    else
      self.output.features:resizeAs(input[1].features):copy(input[1].features)
      self.output.metadata=input[1].metadata
      self.output.spatialSize=input[1].spatialSize
    end
    for i=2,#input do
      assert(input[i].nActive==input[1].nActive)
      self.output.features:narrow(2,1,input[i].features:size(2)):add(input[i].features)
    end
    return self.output
  end

  function CAddTable:updateGradInput(input, gradOutput)
    for i=1,#input do
      if self.inplace and input[1].features:size(2) == input[i].features:size(2) then
        self.gradInput[i]=self.gradInput[i] or {}
        self.gradInput[i].features=gradOutput.features
      else
        self.gradInput[i]=self.gradInput[i] or {features=input[i].features.new()}
        self.gradInput[i].features:resizeAs(input[i].features)
        self.gradInput[i].features:copy(
          gradOutput.features:narrow(2,1,input[i].features:size(2)))
      end
    end
    for i=#input+1,#self.gradInput do
      self.gradInput[i]=nil
    end
    return self.gradInput
  end
  function CAddTable:backwards(input, gradOutput)
    for i=1,#input do
      if self.inplace and input[1].features:size(2) == input[i].features:size(2) then
        self.gradInput[i]=self.gradInput[i] or {}
        self.gradInput[i].features=gradOutput.features
      else
        self.gradInput[i]=self.gradInput[i] or {features=input[i].features.new()}
        self.gradInput[i].features:resizeAs(input[i].features)
        self.gradInput[i].features:copy(
          gradOutput.features:narrow(2,1,input[i].features:size(2)))
      end
    end
    for i=#input+1,#self.gradInput do
      self.gradInput[i]=nil
    end
    return self.gradInput
  end

  function CAddTable:clearState()
    self.gradInput = {}
    self.output = self.inplace and 'recycle' or {
      features = self.output.features:set()
    }
  end

  function CAddTable:suggestInputSize(nOut)
    return nOut
  end
end
