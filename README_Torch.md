## Hello World - (Lua)Torch
Convolutional networks are built with SparseConvNet in the same way as with Torch's nn/cunn/cudnn packages.
```
--Train on the GPU if there is one, otherwise CPU
scn=require 'sparseconvnet'
tensorType = scn.cutorch and 'torch.CudaTensor' or 'torch.FloatTensor'


model = scn.Sequential()
:add(scn.SparseVggNet(2,1,{ --dimension 2, 1 input plane
      {'C', 8}, -- 3x3 VSC convolution, 8 output planes, batchnorm, ReLU
      {'C', 8}, -- and another
      {'MP', 3, 2}, --max pooling, size 3, stride 2
      {'C', 16}, -- etc
      {'C', 16},
      {'MP', 3, 2},
      {'C', 24},
      {'C', 24},
      {'MP', 3, 2}}))
:add(scn.Convolution(2,24,32,3,1,false)) --an SC convolution on top
:add(scn.BatchNormReLU(32))
:add(scn.SparseToDense(2))
:type(tensorType)

--[[
To use the network we must create an scn.InputBatch with right dimensionality.
If we want the output to have spatial size 10x10, we can find the appropriate
input size, give that we uses three layers of MP3/2 max-pooling, and finish
with a SC convoluton
]]

inputSpatialSize=model:suggestInputSize(torch.LongTensor{10,10}) --103x103
input=scn.InputBatch(2,inputSpatialSize)

--Now we build the input batch, sample by sample, and active site by active site.
msg={
  " O   O  OOO  O    O    OO     O       O   OO   OOO   O    OOO   ",
  " O   O  O    O    O   O  O    O       O  O  O  O  O  O    O  O  ",
  " OOOOO  OO   O    O   O  O    O   O   O  O  O  OOO   O    O   O ",
  " O   O  O    O    O   O  O     O O O O   O  O  O  O  O    O  O  ",
  " O   O  OOO  OOO  OOO  OO       O   O     OO   O  O  OOO  OOO   ",
}

input:addSample()
for y,line in ipairs(msg) do
  for x = 1,string.len(line) do
    if string.sub(line,x,x) == 'O' then
      local location = torch.LongTensor{x,y}
      local featureVector = torch.FloatTensor{1}
      input:setLocation(location,featureVector,0)
    end
  end
end

--[[
Optional: allow metadata preprocessing to be done in batch preparation threads
to improve GPU utilization.

Parameter:
3 if using MP3/2 or size-3 stride-2 convolutions for downsizing,
2 if using MP2
]]
input:precomputeMetadata(3)

model:evaluate()
input:type(tensorType)
output = model:forward(input)

--[[
Output is 1x32x10x10: our minibatch has 1 sample, the network has 32 output
feature planes, and 10x10 is the spatial size of the output.
]]
print(output:size())
```


## Torch Setup

Tested with Ubuntu 16.04.
Install [Torch](http://torch.ch/docs/getting-started.html) then: <br />
```
apt-get install libsparsehash-dev
git clone git@github.com:facebookresearch/SparseConvNet.git

then

cd SparseConvNet/Torch/
luarocks make sparseconvnet-0.1-1.rockspec
```
To run the examples you may also need to install unrar and TorchNet:
```
apt-get install unrar

and

luarocks install torchnet
```
