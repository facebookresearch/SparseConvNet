-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local ConcatTable, parent = torch.class(
    'sparseconvnet.ConcatTable', 'nn.ConcatTable', sparseconvnet)

  function ConcatTable:__init()
    parent.__init(self)
    self.modules={}
    self.output={}
    self.gradInput={
      features=torch.Tensor()
    }
    sparseconvnet.shareShared(self)
  end

  function ConcatTable:add(module)
    table.insert(self.modules,module)
    sparseconvnet.shareShared(self)
    return self
  end

  function ConcatTable:updateOutput(input)
    for i = 1,#self.modules do
      self.output[i]=self.modules[i]:forward(input)
    end
    for i = #self.modules+1,#self.output do
      self.output[i]=nil
    end
    return self.output
  end

  function ConcatTable:backward(input, gradOutput)
    local gradInputs={}
    for i = 1,#self.modules do
      gradInputs[i]=self.modules[i]:backward(input,gradOutput[i],scale)
    end
    self.gradInput.features:resizeAs(
      gradInputs[1].features):copy(gradInputs[1].features)
    for i=2,#self.modules do
      self.gradInput.features:add(gradInputs[i].features)
    end
    return self.gradInput
  end

  function ConcatTable:clearState()
    for _,m in ipairs(self.modules) do
      m:clearState()
    end
    self.output={}
    self.gradInput={features=self.gradInput.features:set()}
  end

  function ConcatTable:suggestInputSize(nOut)
    return self.modules[1]:suggestInputSize(nOut)
  end
end
