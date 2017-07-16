-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

-- ModelNet-40 data - https://github.com/charlesq34/3dcnn.torch
-- input is a list of active coordinates in a box [0,29]^3

tnt=require 'torchnet'
scn=require 'sparseconvnet'
require 'paths'

if not paths.dirp('t7') then
  print('Downloading and preprocessing data ...')
  dofile('process.lua')
end

function train(spatialSize,precomputeStride)
  local d={}
  for x=1,590580 do
    d[x]=x
  end
  d=tnt.ListDataset(d,function(x) return torch.load('t7/train/'..x..'.t7') end):shuffle()
  function d:manualSeed(seed) torch.manualSeed(seed) end
  d=tnt.BatchDataset(d,100,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(3,spatialSize)
      local offset = spatialSize/2-15
      local v=torch.FloatTensor({1})
      for _,obj in ipairs(tbl.input) do
        input:addSample()
        obj=obj:type('torch.LongTensor'):add((offset+torch.LongTensor(3):random(-2,2)):view(1,3):expandAs(obj))
        for i=1,obj:size(1) do
          local p = obj[i]
          input:setLocation(obj[i],v,0)
        end
      end
      input:precomputeMetadata(precomputeStride)
      return {input=input,target=torch.LongTensor(tbl.target)}
    end
  )
  d=tnt.ParallelDatasetIterator({
      init = function() require 'torchnet'; scn=require 'sparseconvnet' end,
      nthread = 10,
      closure = function() return d end,
      ordered = true})
  return function(epoch)
    d:exec('manualSeed', epoch)
    d:exec('resample')
    return d()
  end
end
function val(spatialSize,precomputeStride)
  local d={}
  for x=1,148080 do
    d[x]=x
  end
  d=tnt.ListDataset(d,function(x) return torch.load('t7/test/'..x..'.t7') end)
  d=tnt.BatchDataset(d,100,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(3,spatialSize)
      local v=torch.FloatTensor({1})
      local offset = (spatialSize/2-15):view(1,3)
      for _,obj in ipairs(tbl.input) do
        input:addSample()
        obj=obj:type('torch.LongTensor'):add(offset:view(1,3):expandAs(obj))
        for i=1,obj:size(1) do
          input:setLocation(obj[i],v,0)
        end
      end
      input:precomputeMetadata(precomputeStride)
      return {input=input,target=torch.LongTensor(tbl.target)}
    end
  )
  d=tnt.ParallelDatasetIterator({
      init = function() require 'torchnet'; scn=require 'sparseconvnet' end,
      nthread = 10,
      closure = function() return d end,
      ordered = true})
  return function()
    return d()
  end
end

return function(...)
  return {train=train(...), val=val(...)}
end
