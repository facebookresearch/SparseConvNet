# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet as scn

# Use the GPU if there is one, otherwise CPU
use_cuda = torch.cuda.is_available()

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
)
if use_cuda:
    model.cuda()

# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
input = scn.InputBatch(2, inputSpatialSize)

msg = [
    " X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
    " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
    " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
    " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
    " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "]

# Add a sample using set_location
input.add_sample()
for y, line in enumerate(msg):
    for x, c in enumerate(line):
        if c == 'X':
            location = torch.LongTensor([y, x])
            featureVector = torch.FloatTensor([1])
            input.set_location(location, featureVector, 0)

# Add a sample using set_locations
input.add_sample()
locations = []
features = []
for y, line in enumerate(msg):
    for x, c in enumerate(line):
        if c == 'X':
            locations.append([y, x])
            features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features)
input.set_locations(locations, features, 0)

# Optional: allow metadata preprocessing to be done in batch preparation threads
# to improve GPU utilization.
#
# Parameter:
#    3 if using MP3/2 pooling or C3/2 convolutions for downsizing,
#    2 if using MP2 pooling for downsizing.
input.precomputeMetadata(3)

model.train()
if use_cuda:
    input.cuda()
output = model.forward(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print(output.shape, output.type())
