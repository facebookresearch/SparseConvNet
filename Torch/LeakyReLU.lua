-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C
  local math = require 'math'
  local LeakyReLU, parent = torch.class(
    'sparseconvnet.LeakyReLU', 'nn.Module', sparseconvnet)

  function LeakyReLU:__init(leakage,ip)
    parent.__init(self)
    self.inplace = type(ip)~='boolean' or ip --default to inplace
    self.leakage = leakage

    self.output = {
      features = ip and "Recycle input.features" or torch.Tensor(),
    }
    self.gradInput = {
      features = ip and "Recycle gradOutput.features" or torch.Tensor()
    }
  end

  function LeakyReLU:updateOutput(input)
    self.output.metadata=input.metadata
    self.output.spatialSize = input.spatialSize
    C.typedFn(self._type,'LeakyReLU_updateOutput')(
      input.features:cdata(),
      self.output.features:cdata(),
      self.leakage)
    return self.output
  end

  function LeakyReLU:updateGradInput(input, gradOutput)
    if self.inplace then
      self.gradInput.features = gradOutput.features
    else
      self.gradInput.features:resizeAs(gradOutput.features)
    end
    C.typedFn(self._type,'LeakyReLU_updateGradInput')(
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.leakage)
    return self.gradInput
  end

  function LeakyReLU:__tostring()
    local s = 'LeakyReLU('..self.leakage..')'
    return s
  end

  function LeakyReLU:clearState()
    self.output = {
      features = self.inplace and "Recycle input.features" or self.output.features:set(),
    }
    self.gradInput = {
      features = self.inplace and "Recycle gradOutput.features" or self.gradInput.features:set()
    }
  end

  function LeakyReLU:suggestInputSize(nOut)
    return nOut
  end
end
