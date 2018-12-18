# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .metadata import Metadata
from .sparseConvNetTensor import SparseConvNetTensor

class DenseToSparse(Module):
    """
    Function to convert a Dense Input into a sparse input.
    If possible, avoid using this module; build the hidden layer using InputBatch.

    Parameters:
    dimension : of the input field
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension

    def forward(self, input):
        output = SparseConvNetTensor()
        output.metadata = Metadata(self.dimension)
        output.spatial_size = torch.LongTensor(list(input.size()[2:]))
        output.features = DenseToSparseFunction.apply(
            input,
            output.metadata,
            output.spatial_size,
            self.dimension)
        return output

    def __repr__(self):
        return 'DenseToSparse(' + str(self.dimension) + ')'

    def input_spatial_size(self, out_size):
        return out_size

class DenseToSparseFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input,
            output_metadata,
            output_spatial_size,
            dimension):
        ctx.dimension = dimension
        aa = input.permute(
            *([0, ] + list(range(2, 2 + dimension)) + [1, ])).clone()
        ctx.aas = aa.size()
        nz = aa.abs().sum(dimension + 1).view(aa.size()[0:-1])
        s = torch.LongTensor(nz.stride()).view(1, dimension + 1)
        nz = nz.nonzero()
        s = s.type_as(nz)
        aa = aa.view(-1, input.size(1))
        ctx.aas2 = aa.size()
        r = (nz * s.expand_as(nz)).sum(1).view(-1)
        output_features = aa.index_select(0, r)
        output_metadata.createMetadataForDenseToSparse(
            output_spatial_size,
            nz.cpu(),
            input.size(0))
        ctx.save_for_backward(output_features, r)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        output_features, r = ctx.saved_tensors
        grad_input = grad_output.new().resize_(
            ctx.aas2).zero_().index_copy_(0, r, grad_output)
        grad_input = grad_input.view(ctx.aas).permute(
            *([0, ctx.dimension + 1] + list(range(1, ctx.dimension + 1))))
        return grad_input, None, None, None
