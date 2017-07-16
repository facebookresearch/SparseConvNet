-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

local threads=require 'threads'
t=threads.Threads(30)
for j=1,30 do
  t:addjob(function()
      function rescaleCharacter(c)
        local cc=torch.cat(c,1)
        local m=cc:min(1)
        local s=(cc:max(1)-m):float()
        for i=1,#c do
          c[i]=(torch.cdiv((c[i]-m:expandAs(c[i])):float(),s:expandAs(c[i]))*255.99):byte()
        end
      end
      for i=j,898573,30 do
        ch=dofile('t7/train/'..i..'.lua')
        os.remove('t7/train/'..i..'.lua')
        rescaleCharacter(ch.input)
        torch.save('t7/train/'..i..'.t7',ch)
      end
      for i=j,224559,30 do
        ch=dofile('t7/test/'..i..'.lua')
        os.remove('t7/test/'..i..'.lua')
        rescaleCharacter(ch.input)
        torch.save('t7/test/'..i..'.t7',ch)
      end
    end
  )
end
for i=1,30 do
  t:dojob()
end
