import sparseconvnet as scn
import torch
from torch.autograd import Variable

input_layer = scn.Sequential()\
            .add(scn.InputLayer(2, 200))\

fusion_layer = scn.GlobalFusion(2, 1.8)


global_locs = [
    [0, 0], [0, 1], [1, 0], [1, 1]
]
global_feats = [
    [.1, .2, .3],
    [.2, .3, .4],
    [.4, .5, .6],
    [.7, .8, .9]
]

'''
#
# #
# # #
'''
local_locs = [
    [0, 0],
    [1, 1],
    [1, 0],
    [2, 0],
    [2, 1],
    [2, 2],
]

local_feats = [
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
]
local_locs = torch.Tensor(local_locs)
local_feats = torch.Tensor(local_feats).cuda()
local_feats = Variable(local_feats, requires_grad=True)
global_locs = torch.Tensor(global_locs)
global_feats = torch.Tensor(global_feats).cuda()
global_feats = Variable(global_feats, requires_grad=True)
x = input_layer([local_locs, local_feats])
y = input_layer([global_locs, global_feats])
print(x)
print(fusion_layer)

local_base = torch.Tensor([[1, 1]])
global_base = torch.Tensor([[1, 1]])
# base.requires_grad_(False)
fuse = fusion_layer(x, y, local_base, global_base)
# print(fuse)
print(fuse.features)
print(fuse.features.backward(fuse.features))
