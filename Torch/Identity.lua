-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local Identity, _ = torch.class(
    'sparseconvnet.Identity', 'nn.Identity', sparseconvnet)

  function Identity:clearState()
    self.output=nil
    self.gradInput=nil
  end

  function Identity:suggestInputSize(nOut)
    return nOut
  end
end
