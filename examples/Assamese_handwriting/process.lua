-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

train={}
test={}
torch.manualSeed(0)
p=torch.randperm(45)

function rescaleCharacter(c)
  local cc=torch.cat(c,1)
  local m=cc:min(1)
  local s=(cc:max(1)-m):float()
  for i=1,#c do
    c[i]=(torch.cdiv((c[i]-m:expandAs(c[i])):float(),s:expandAs(c[i]))*255.99):byte()
  end
  return c
end

for char = 1,183 do
  for writer = 1,36 do
    train[#train+1]={input=rescaleCharacter(dofile('tmp/' .. char .. '.' .. p[writer] .. '.lua')),target=char}
  end
end
for char = 1,183 do
  for writer = 37,45 do
    test[#test+1]={input=rescaleCharacter(dofile('tmp/' .. char .. '.' .. p[writer] .. '.lua')),target=char}
  end
end
print(#train,#test)
os.execute('mkdir t7/')
torch.save('t7/train.t7',train)
torch.save('t7/test.t7',test)
