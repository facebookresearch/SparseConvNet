-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Implementation of batchwise dropout, optionally followed by LeakyReLU

Parameters:
p : dropout probability in the range [0,1]
ip : perform dropout inplace (default true)
leaky : in the range [0,1]. Set to zero to do ReLU after the dropout. Set to one
just to do dropout. Set to 1/3 for LeakyReLU after the dropout, etc. (default 1)
]]

return function(sparseconvnet)
  local C = sparseconvnet.C
  local math = require 'math'
  local BatchwiseDropout, parent = torch.class(
    'sparseconvnet.BatchwiseDropout', 'nn.Module', sparseconvnet)

  function BatchwiseDropout:__init(nPlanes,p,ip,leaky)
    parent.__init(self)
    self.inplace = (type(ip)~='boolean') or ip
    self.p = p
    self.leakiness=leaky or 1
    self.noise=torch.Tensor(nPlanes)
    self.nPlanes=nPlanes
    self.output = ip and "Recycle" or {
      features = torch.Tensor()
    }
    self.gradInput = ip and "Recycle" or {
      features = torch.Tensor()
    }
  end

  function BatchwiseDropout:updateOutput(input)
    if self.train then
      self.noise:bernoulli(1-self.p)
    else
      self.noise:fill(1-self.p)
    end
    if self.inplace then
      self.output = input
    else
      self.output.metadata = input.metadata
      self.output.spatialSize = input.spatialSize
    end
    C.typedFn(self._type,'BatchwiseMultiplicativeDropout_updateOutput')(
      input.features:cdata(),
      self.output.features:cdata(),
      self.noise:cdata(),
      self.leakiness)
    return self.output
  end

  function BatchwiseDropout:updateGradInput(input, gradOutput)
    if self.inplace then
      self.gradInput = gradOutput
    end
    C.typedFn(self._type,'BatchwiseMultiplicativeDropout_updateGradInput')(
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.noise:cdata(),
      self.leakiness)
    return self.gradInput
  end

  function BatchwiseDropout:type(type)
    self._type=type
    self.noise=self.noise:type(type)
    if self.output.features then
      self.output.features=self.output.features:type(type)
    end
    if self.gradInput.features then
      self.gradInput.features=self.gradInput.features:type(type)
    end
  end

  function BatchwiseDropout:__tostring()
    local s = 'BatchwiseDropout('..self.p .. ", " .. self.leakiness..')'
    return s
  end

  function BatchwiseDropout:clearState()
    if self.inplace then
      self.output=nil
      self.gradInput=nil
    else
      self.output={features=self.output.features:set()}
      self.gradInput={features=self.gradInput.features:set()}
    end
  end

  function BatchwiseDropout:suggestInputSize(nOut)
    return nOut
  end
end
