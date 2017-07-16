-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

-- Borrow code (lines 37-67) from https://github.com/charlesq34/3dcnn.torch/blob/master/provider.lua

--[[
3dcnn.torch (Volumetric ConvNets)

Copyright (c) 2016, Geometric Computation Group of Stanford University

The MIT License (MIT)

Copyright (c) 2016 Charles R. Qi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

require 'paths'
require 'hdf5'

-- data from https://github.com/charlesq34/3dcnn.torch
-- download dataset 60x azimuth+elevation augmented
if not paths.dirp('data/modelnet40_60x') then
  local www = 'https://shapenet.cs.stanford.edu/media/modelnet40_h5.tar'
  local tar = paths.basename(www)
  os.execute('mkdir data')
  os.execute('wget ' .. www .. '; ' .. 'tar xvf ' .. tar)
  os.execute('mv modelnet40_* data')
end

function getDataFiles(input_file)
    local train_files = {}
    for line in io.lines(input_file) do
        train_files[#train_files+1] = line
    end
    return train_files
end

-- load h5 file data into memory
function loadDataFile(file_name)
  print(paths.filep(file_name))
    local current_file = hdf5.open(file_name,'r')
    local current_data = current_file:read('data'):all():float()
    current_data[current_data:eq(2)] = 1 --convert to binary occupancy
    local current_label = torch.squeeze(current_file:read('label'):all():add(1))
    current_file:close()
    return current_data, current_label
end


--[[
Copyright 2016-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
]]

os.execute('mkdir -p t7/train/')
train_files=getDataFiles('data/modelnet40_60x/train_data.txt')
n=1
for fn = 1, #train_files do
  current_data, current_label = loadDataFile(train_files[fn])
  print(current_data:size(),current_label:size())
  current_data:resize(current_data:size(1),30,30,30)
  for j=1,current_data:size(1) do
    nz=current_data[j]:nonzero():csub(1):type('torch.ByteTensor')
    torch.save('t7/train/'..n..'.t7',{input=nz,target=current_label[j]})
    n=n+1
  end
end

os.execute('mkdir -p t7/test/')
test_files=getDataFiles('data/modelnet40_60x/test_data.txt')
n=1
for fn = 1, #test_files do
  current_data, current_label = loadDataFile(test_files[fn])
  print(current_data:size(),current_label:size())
  current_data:resize(current_data:size(1),30,30,30)
  for j=1,current_data:size(1) do
    nz=current_data[j]:nonzero():csub(1):type('torch.ByteTensor')
    torch.save('t7/test/'..n..'.t7',{input=nz,target=current_label[j]})
    n=n+1
  end
end
