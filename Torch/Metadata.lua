-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Store Metadata relating to which spatial locations are active at each scale.
Convolutions and 'convolution reversing' deconvolutions
all coexist within the same MetaData object as long as each spatial size
only occurs once. Valid convolutions do not change the spatial structure.

Serialization is emulated by storing the pointer as an integer.
This is sufficient for mutithreaded batch preparation: each 'serialized'
object must be de-serialized exactly once.
]]

return function(sparseconvnet)
  local ffi=require 'ffi'
  local C = sparseconvnet.C
  local Metadata = torch.class('sparseconvnet.Metadata', sparseconvnet)
  function Metadata:__init(dimension)
    self.ffi=ffi.new('void *[1]')
    self.dimension=dimension
    --[[ffi.gc to delete Metadata that need to be garbage collected.]]
    ffi.gc(self.ffi, C.dimensionFn(self.dimension, 'freeMetadata'))
  end
  function Metadata:deactivateGC()
    --[[FFI pointers cannot be serialized for passing from batch-creation
    worker threads to the main learner thread. So we convert it to something
    that can, ...]]
    if self.ffi then
      self.p = torch.LongStorage(1)
      C.copyFfiPtrToLong(self.p,self.ffi)
      ffi.gc(self.ffi,nil)
      self.ffi=nil
    end
  end
  function Metadata:reactivateGC()
    --[[... and then convert them back as soon as possible after the thread
    to thread transfer, and hope no-one notices.]]
    if self.p then
      self.ffi=ffi.new('void *[1]')
      ffi.gc(self.ffi, C.dimensionFn(self.dimension, 'freeMetadata'))
      C.copyLongToFfiPtr(self.ffi,self.p)
      self.p=nil
    end
  end
  function Metadata:write(file)
    self:deactivateGC()
    file:writeObject({self.dimension,self.p})
  end
  function Metadata:read(file)
    local r = file:readObject()
    self.dimension=r[1]
    self.p=r[2]
    self:reactivateGC()
  end
end
