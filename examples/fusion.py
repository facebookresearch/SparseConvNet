import sparseconvnet as scn
import torch

input_layer = scn.Sequential()\
            .add(scn.InputLayer(2, 200))\

fusion_layer = scn.GlobalFusion(2, 1.8)

global_locs = [
    [0, 0], [0, 1], [1, 0], [1, 1], [10, 10]
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
local_feats = torch.Tensor(local_feats)
global_locs = torch.Tensor(global_locs)
global_feats = torch.Tensor(global_feats)
x = input_layer([local_locs, local_feats])
y = input_layer([global_locs, global_feats])
print(x)

base = torch.Tensor([[0, 0, 0]])
base.requires_grad_(False)
fuse = fusion_layer(x, y, base, base)
print(fuse)
