-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local C = sparseconvnet.C
  local AF,parent = torch.class(
    'sparseconvnet.AffineReluTrivialConvolution', 'nn.Module',sparseconvnet)

  function AF:__init(nInputPlanes, nOutputPlanes,additiveGrad)
    parent.__init(self)
    self.nInputPlanes=nInputPlanes
    self.nOutputPlanes=nOutputPlanes
    self.affineWeight = torch.Tensor(nInputPlanes)
    self.affineBias = torch.Tensor(nInputPlanes)
    self.convWeight = torch.Tensor(nInputPlanes,nOutputPlanes)
    self.gradAffineWeight = torch.Tensor(nInputPlanes):fill(0)
    self.gradAffineBias = torch.Tensor(nInputPlanes):zero()
    self.gradConvWeight = torch.Tensor(nInputPlanes,nOutputPlanes):zero()
    self.additiveGrad=additiveGrad or false --boolean
    self.output={
      features=torch.Tensor(),
    }
    self:reset()
  end

  function AF:reset()
    self.affineWeight:fill(1)
    self.affineBias:zero()
    self.convWeight:normal(0,math.sqrt(2/self.nInputPlanes)) --not 2/self.nOutputPlanes?
  end

  function AF:parameters()
    return {self.affineWeight, self.affineBias, self.convWeight},
    {self.gradAffineWeight, self.gradAffineBias, self.gradConvWeight}
  end

  function AF:updateOutput(input)
    self.output.metadata = input.metadata
    self.output.spatialSize = input.spatialSize
    C.typedFn(self._type,'AffineReluTrivialConvolution_updateOutput')(
      input.features:cdata(),
      self.output.features:cdata(),
      self.affineWeight:cdata(),
      self.affineBias:cdata(),
      self.convWeight:cdata())
    self.shared.forwardPassMultiplyAddCount=
    self.shared.forwardPassMultiplyAddCount+
    input.features:size(1)*self.nInputPlanes*self.nOutputPlanes
    self.shared.forwardPassHiddenStates=
    self.shared.forwardPassHiddenStates+self.output.features:nElement()
    return self.output
  end

  function AF:backward(input, gradOutput)
    C.typedFn(self._type,'AffineReluTrivialConvolution_backward')(
      input.features:cdata(),
      self.gradInput.features:cdata(),
      gradOutput.features:cdata(),
      self.affineWeight:cdata(),
      self.gradAffineWeight:cdata(),
      self.affineBias:cdata(),
      self.gradAffineBias:cdata(),
      self.convWeight:cdata(),
      self.gradConvWeight:cdata(),
      self.additiveGrad)
    return self.gradInput
  end

  function AF:updateGradInput(input, gradOutput)
    assert(false) --just call backward
  end

  function AF:accGradParameters(input, gradOutput, scale)
    assert(false) --just call backward
  end

  function AF:__tostring()
    local s = 'AffineReluTrivialConvolution(' ..
    self.nInputPlanes..'->' .. self.nOutputPlanes .. ')'
    return s
  end

  function AF:clearState()
    self.output={features=self.output.features:set()}
    self.gradInput={features=self.gradInput.features:set()}
    self.rules=nil
  end

  function AF:suggestInputSize(nOut)
    return nOut
  end
end
