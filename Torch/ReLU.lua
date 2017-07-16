-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Parameters
ip : operate in place (default true)
]]

return function(sparseconvnet)
  local ReLU, parent = torch.class(
    'sparseconvnet.ReLU', 'sparseconvnet.LeakyReLU', sparseconvnet)

  function ReLU:__init(ip)
    parent.__init(self,0,ip)
  end
end
