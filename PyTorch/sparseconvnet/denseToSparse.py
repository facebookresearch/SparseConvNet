# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Function to convert a Dense Input into a sparse input.
If possible, avoid using this module; build the hidden layer using InputBatch.

Parameters:
dimension : of the input field
"""

from torch.autograd import Function, Variable
from torch.nn import Module
from .utils import *
from .metadata import Metadata
from .sparseConvNetTensor import SparseConvNetTensor

class DenseToSparseFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input,
            output_metadata,
            output_spatial_size,
            dimension):
        ctx.dimension=dimension
        a=input
        aa=a.permute(*([0,]+list(range(2,2+dimension))+[1,])).clone()
        ctx.aas=aa.size()
        nz=aa.abs().sum(dimension+1).view(aa.size()[0:-1])
        s=torch.LongTensor(nz.stride()).view(1,dimension+1)
        nz=nz.nonzero()
        s=s.type_as(nz)
        aa=aa.view(-1,a.size(1))
        ctx.aas2=aa.size()
        ctx.r=(nz*s.expand_as(nz)).sum(1).view(-1)
        ctx.output_features=aa.index_select(0,ctx.r)
        dim_fn(dimension, 'createMetadataForDenseToSparse')(
            output_metadata.ffi,
            output_spatial_size,
            nz.cpu(),
            input.size(0))
        return ctx.output_features
    @staticmethod
    def backward(ctx, grad_output):
        grad_input=Variable(grad_output.data.new().resize_(ctx.aas2).zero_().index_copy_(0,ctx.r,grad_output.data))
        grad_input=grad_input.view(ctx.aas).permute(*([0,ctx.dimension+1]+list(range(1,ctx.dimension+1))))
        return grad_input, None, None, None



class DenseToSparse(Module):
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension

    def forward(self, input):
        output = SparseConvNetTensor()
        output.metadata = Metadata(self.dimension)
        output.spatial_size=torch.LongTensor(list(input.size()[2:]))
        output.features=DenseToSparseFunction().apply(
            input,
            output.metadata,
            output.spatial_size,
            self.dimension)
        return output

    def __repr__(self):
        return 'DenseToSparse(' + str(self.dimension) + ')'

    def input_spatial_size(self, out_size):
        return out_size
