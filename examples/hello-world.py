# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet as scn


# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")

model = scn.Sequential().add(
    scn.SparseVggNet(2, 1,
                     [['C', 8], ['C', 8], ['MP', 3, 2],
                      ['C', 16], ['C', 16], ['MP', 3, 2],
                      ['C', 24], ['C', 24], ['MP', 3, 2]])
).add(
    scn.SubmanifoldConvolution(2, 24, 32, 3, False)
).add(
    scn.BatchNormReLU(32)
).add(
    scn.SparseToDense(2, 32)
).to(device)

# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
input_layer = scn.InputLayer(2, inputSpatialSize)
bl_input_layer = scn.BLInputLayer(2, inputSpatialSize)

msgs = [[" X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
         " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
         " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
         " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
         " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],

        [" XXX              XXXXX      x   x     x  xxxxx  xxx ",
         " X  X  X   XXX       X       x   x x   x  x     x  x ",
         " XXX                X        x   xxxx  x  xxxx   xxx ",
         " X     X   XXX       X       x     x   x      x    x ",
         " X     X          XXXX   x   x     x   x  xxxx     x ",]]


# Create Nx3 and Nx1 vectors to encode the messages above using InputLayer:
locations = []
features = []
for batchIdx, msg in enumerate(msgs):
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                locations.append([y, x, batchIdx])
                features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features).to(device)

input = input_layer([locations,features])
print('Input SparseConvNetTensor:', input)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)




# Alternatively:
# Create Nx3 and Nx1 vectors to encode the messages above using BLInputLayer:
batch=[]
for batchIdx, msg in enumerate(msgs):
    l,f=[],[]
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                l.append([y, x])  #Locations
                f.append([1])     #Features
    batch.append([torch.LongTensor(l),torch.FloatTensor(f)])
batch=scn.prepare_BLInput(batch)
batch[1]=batch[1].to(device)
    
input = bl_input_layer(batch)
print('Input SparseConvNetTensor:', input)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)


