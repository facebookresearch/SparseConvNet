-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Assume all the inputs have identical SparseGrids and input[i].features:size(1)
Assume input[1].features:size(2) >= input[i].features:size(2) for all i=1,#input
output.validRules is taken from input[1].validRules (could do set union?)
(for resnets, make sure the residual link is input[2])
]]

return function(sparseconvnet)
  local JoinTable, parent = torch.class(
    'sparseconvnet.JoinTable', 'nn.Module', sparseconvnet)

  function JoinTable:__init(nPlanes)
    parent.__init(self)
    self.nOutputPlanes=0
    self.nPlanes=nPlanes
    self.gradInput = {}
    for i,j in ipairs(nPlanes) do
      self.gradInput[i]={features=torch.Tensor()}
      self.nOutputPlanes=self.nOutputPlanes+j
    end
    self.output = {
      features = torch.Tensor()
    }
  end

  function JoinTable:updateOutput(input)
    self.output.features:resize(input[1].features:size(1),self.nOutputPlanes)
    self.output.metadata=input[1].metadata
    self.output.spatialSize=input[1].spatialSize
    local offset=0
    for i,j in ipairs(self.nPlanes) do
      assert(input[i].features:size(1)==input[1].features:size(1))
      assert(input[i].features:size(2)==j)
      self.output.features:narrow(2,1+offset,j):copy(input[i].features)
      offset=offset+j
    end
    return self.output
  end

  function JoinTable:updateGradInput(input, gradOutput)
    local offset=0
    for i,j in ipairs(self.nPlanes) do
      self.gradInput[i].features:resize(input[1].features:size(1),j):copy(gradOutput.features:narrow(2,1+offset,j))
      offset=offset+j
    end
    return self.gradInput
  end

  function JoinTable:clearState()
    self.output = {
      features = self.output.features:set()
    }
    for i,j in ipairs(self.nPlanes) do
      self.gradInput[i]={features=self.gradInput[i].features:set()}
    end
  end

  function JoinTable:suggestInputSize(nOut)
    return nOut
  end
end
