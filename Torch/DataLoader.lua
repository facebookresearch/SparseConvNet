-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--Based on https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
return function(sparseconvnet)
  local DataLoader = torch.class('sparseconvnet.DataLoader', sparseconvnet)
  function DataLoader:__init(data, nSamples, batchSize, nThreads,
      encode)
    local threads = require 'threads'
    self.nSamples = nSamples
    self.batchSize = batchSize
    self.nThreads=nThreads

    if nThreads>0 then
      self.threads=threads.Threads(nThreads,
        function(threadid)
          g_data=data
          g_encode=encode
          torch.manualSeed(torch.random())
        end)
      function self:epoch()
        local perm=torch.randperm(self.nSamples)
        local idx,sample = 1, nil
        local function enqueue()
          while idx <= self.nSamples and self.threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(
                self.batchSize, self.nSamples - idx + 1))
            self.threads:addjob(
              function(indices)
                require 'nn'
                batch=g_encode(g_data,indices:clone())
                collectgarbage()
                return batch
              end,
              function(batch)
                require 'nn'
                sample=batch
              end,
              indices)
            idx = idx + self.batchSize
          end
        end
        local function loop()
          enqueue()
          if not self.threads:hasjob() then
            return nil
          end
          self.threads:dojob()
          enqueue()
          return sample
        end
        return loop
      end
    else --nThreads==0, for debugging
      self._data=data
      self._encode=encode
      self._postSerialize=postSerialize or function (batch) end
      self._postEpoch=postEpoch or function () end
      function self:epoch()
        perm=torch.randperm(self.nSamples)
        local idx,sample = 1, nil
        local function loop()
          if idx <= self.nSamples then
            local indices = perm:narrow(1, idx, math.min(
                self.batchSize, self.nSamples - idx + 1))
            batch=self._encode(self._data,indices:clone())
            collectgarbage()
            sample=batch
            idx = idx + self.batchSize
            return sample
          else
            return nil
          end
        end
        return loop
      end
    end
  end
end
