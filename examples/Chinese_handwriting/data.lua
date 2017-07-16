-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

-- CASIA-OLHWDB1.1 dataset, 3755 character classes
-- 898573 training samples (240 writers), 224559 test samples (60 writers)
--data from http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

-- Consider moving the data folder to a ramdisk, i.e.
-- mkdir /ramdisk
-- mount -t tmpfs -o size=5g tmpfs /ramdisk
-- rsync -r t7/ /ramdisk/t7/
-- change t7/ to /ramdisk/t7/ below (twice)

tnt=require 'torchnet'
scn=require 'sparseconvnet'
require 'paths'

if not paths.dirp('t7/') then
  if not paths.filep('OLHWDB1.1trn_pot.zip') then
    print('Downloading data ...')
    os.execute('wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip')
    os.execute('wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip')
    os.execute('mkdir -p t7/train/ t7/test/ POT/')
    os.execute('unzip OLHWDB1.1trn_pot.zip -d POT/')
    os.execute('unzip OLHWDB1.1tst_pot.zip -d POT/')
  end
  print('Preprocessing data ...')
  os.execute('python3 readPotFiles.py')
  dofile('process.lua')
end

function train(spatialSize,Scale,precomputeStride)
  local d={}
  for x=1,898573
  do
    d[x]=x
  end
  d=tnt.ListDataset(d,function(x) return torch.load('t7/train/'..x..'.t7') end):shuffle()
  function d:manualSeed(seed) torch.manualSeed(seed) end
  d=tnt.BatchDataset(d,100,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(2,spatialSize)
      local center=spatialSize:float():view(1,2)/2
      local p=torch.LongTensor(2)
      local v=torch.FloatTensor({1,0,0})
      for _,char in ipairs(tbl.input) do
        input:addSample()
        for _,stroke in ipairs(char) do
          stroke=stroke:float()/255-0.5
          stroke=center:expandAs(stroke)+stroke*(Scale-0.01)
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
  local d={}
  for x=1,224559
  do
    d[x]=x
  end
  d=tnt.ListDataset(d,function(x) return torch.load('t7/test/'..x..'.t7') end)
  d=tnt.BatchDataset(d,100,function(idx, size) return idx end,function (tbl)
      input=scn.InputBatch(2,spatialSize)
      local center=spatialSize:float():view(1,2)/2
      local p=torch.LongTensor(2)
      local v=torch.FloatTensor({1,0,0})
      for _,char in ipairs(tbl.input) do
        input:addSample()
        for _,stroke in ipairs(char) do
          stroke=stroke:float()/255-0.5
          stroke=center:expandAs(stroke)+stroke*(Scale-0.01)
          scn.C.dimensionFn(2,'drawCurve')(
            input.metadata.ffi,input.features:cdata(),stroke:cdata())
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
