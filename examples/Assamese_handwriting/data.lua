-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

tnt=require 'torchnet'
scn=require 'sparseconvnet'
require 'paths'

if not paths.dirp('t7/') then
  print('Downloading and preprocessing data ...')
  os.execute('bash process.sh')
  dofile('process.lua')
end

function train(spatialSize,Scale,precomputeStride)
  local d=torch.load('t7/train.t7')
  print('Replicating training set 10 times (1 epoch = 10 iterations through the training set = 10x6588 training samples)')
  for i=1,9 do
    for j=1,6588 do
      d[#d+1]=d[j]
    end
  end
  d=tnt.ListDataset(d,function(x) return x end):shuffle()
  function d:manualSeed(seed) torch.manualSeed(seed) end
  d=tnt.BatchDataset(d,108,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(2,spatialSize)
      local center=spatialSize:float():view(1,2)/2
      local p=torch.LongTensor(2)
      local v=torch.FloatTensor({1,0,0})
      for _,char in ipairs(tbl.input) do
        input:addSample()
        local m=torch.eye(2):float()
        local r=torch.random(3)
        local alpha=torch.uniform(-0.2,0.2)
        if alpha==1 then
          m[1][2]=alpha
        elseif alpha==2 then
          m[2][1]=alpha
        else
          m=torch.mm(m,torch.FloatTensor({
                {math.cos(alpha),math.sin(alpha)},
                {-math.sin(alpha),math.cos(alpha)}}))
        end
        c=(center+torch.FloatTensor(1,2):uniform(-8,8))
        for _,stroke in ipairs(char) do
          stroke=stroke:float()/255-0.5
          stroke=c:expandAs(stroke)+stroke*m*(Scale-0.01)
          --------------------------------------------------------------
          -- Draw pen stroke:
          scn.C.dimensionFn(2,'drawCurve')(
            input.metadata.ffi,input.features:cdata(),stroke:cdata())
          --------------------------------------------------------------
          -- Above is equivalent to :
          -- local x1,x2,y1,y2,l=0,stroke[1][1],0,stroke[1][2],0
          -- for i=2,stroke:size(1) do
          -- x1=x2
          -- y1=y2
          -- x2=stroke[i][1]
          -- y2=stroke[i][2]
          -- l=1e-10+((x2-x1)^2+(y2-y1)^2)^0.5
          -- v[2]=(x2-x1)/l
          -- v[3]=(y2-y1)/l
          -- l=math.max(x2-x1,y2-y1,x1-x2,y1-y2,0.9)
          -- for j=0,1,1/l do
          -- p[1]=math.floor(x1*j+x2*(1-j))
          -- p[2]=math.floor(y1*j+y2*(1-j))
          -- input:setLocation(p,v,false)
          -- end
          -- end
          --------------------------------------------------------------
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
function val(spatialSize,Scale,precomputeStride)
  local d=torch.load('t7/test.t7')
  d=tnt.ListDataset(d,function(x) return x end)
  d=tnt.BatchDataset(d,183,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(2,spatialSize)
      local center=spatialSize:float():view(1,2)/2
      local p=torch.LongTensor(2)
      local v=torch.FloatTensor({1,0,0})
      for _,char in ipairs(tbl.input) do
        input:addSample()
        for _,stroke in ipairs(char) do
          stroke=stroke:float()/255-0.5
          stroke=center:expandAs(stroke)+stroke*(Scale-0.01)
          scn.C.dimensionFn(2,'drawCurve')(input.metadata.ffi,input.features:cdata(),stroke:cdata())
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
