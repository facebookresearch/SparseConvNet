-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Child class of the nn.Sequential ConvNet module container.
1. Fill it up with modules e.g. convolutions, max-pooling, ConcatTables,
CAddTables, Sequentials, ...
2. Use :suggestInputSize to determine how large the the spatial size of the
incoming InputBatches should be to get a desired output spatial size, i.e to
produce output of size 7x7, the input to a 2x2 Max-Pooling layer should be
14x14, and so on inductively, backwards through the network.
]]

return function(sparseconvnet)
  local Sequential, parent = torch.class(
    'sparseconvnet.Sequential', 'nn.Sequential', sparseconvnet)

  function Sequential:__init(...)
    parent.__init(self, ...)
    sparseconvnet.shareShared(self)
  end

  function Sequential:add(module)
    table.insert(self.modules,module)
    sparseconvnet.shareShared(self)
    return self
  end

  function Sequential:updateOutput(input)
    local currentOutput = input
    if input.precomputed then
      self.hasSeenPrecomputedInput = true
      self.shared.precomputed=input.precomputed
    elseif self.hasSeenPrecomputedInput then
      self.shared.precomputed=nil
    end
    for i=1,#self.modules do
      currentOutput = self:rethrowErrors(
        self.modules[i], i, 'updateOutput', currentOutput)
    end
    self.output = currentOutput
    return currentOutput
  end

  function Sequential:updateGradInput(input, gradOutput)
    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(
        currentModule, i+1, 'updateGradInput',
        previousModule.output, currentGradOutput)
      currentModule = previousModule
    end
    currentGradOutput = self:rethrowErrors(
      currentModule, 1, 'updateGradInput', input, currentGradOutput)
    self.gradInput = currentGradOutput
    return currentGradOutput
  end

  function Sequential:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(
        currentModule, i+1, 'accGradParameters',
        previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
    end
    self:rethrowErrors(currentModule, 1, 'accGradParameters', input,
      currentGradOutput, scale)
  end

  function Sequential:backward(input, gradOutput, scale)
    scale = scale or 1
    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(currentModule, i+1, 'backward',
        previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
    end
    currentGradOutput = self:rethrowErrors(currentModule, 1, 'backward', input,
      currentGradOutput, scale)
    self.gradInput = currentGradOutput
    return currentGradOutput
  end

  function Sequential:accUpdateGradParameters(input, gradOutput, lr)
    local currentGradOutput = gradOutput
    local currentModule = self.modules[#self.modules]
    for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(currentModule, i+1, 'accUpdateGradParameters',
        previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
    end

    self:rethrowErrors(currentModule, 1, 'accUpdateGradParameters', input,
      currentGradOutput, lr)
  end

  function Sequential:type(tensortype)
    self._type=tensortype
    for i=1,#self.modules do
      self.modules[i]:type(tensortype)
    end
    sparseconvnet.shareShared(self)
    return self
  end

  function Sequential:clearState()
    self.shared.precomputed=nil
    if self.shared.rulesBuffer then
      self.shared.rulesBuffer:set()
    end
    self.output=nil
    self.gradInput=nil
    for _,m in pairs(self.modules) do
      m:clearState()
    end
  end

  function Sequential:suggestInputSize(n)
    for i = #self.modules,1,-1 do
      n=self.modules[i]:suggestInputSize(n)
    end
    return n
  end
end
