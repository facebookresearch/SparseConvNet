# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Function to convert a SparseConvNet hidden layer to a dense convolutional
layer. Put a SparseToDense convolutional layer (or an ActivePooling layer) at
the top of your sparse network. The output can then pass to a dense
convolutional layers or (if the spatial dimensions have become trivial) a
linear classifier.

Parameters:
dimension : of the input field,
"""

from torch.autograd import Function, Variable
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class SparseToDenseFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            input_metadata,
            spatial_size,
            dimension,
            nPlanes):
        ctx.input_metadata=input_metadata
        ctx.spatial_size=spatial_size
        ctx.dimension=dimension
        ctx.input_features=input_features
        output = input_features.new()
        dim_typed_fn(ctx.dimension, input_features, 'SparseToDense_updateOutput')(
            spatial_size,
            input_metadata.ffi,
            input_features,
            output,
            torch.cuda.IntTensor() if input_features.is_cuda else nullptr,
            nPlanes)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input=Variable(grad_output.data.new())
        dim_typed_fn(ctx.dimension, ctx.input_features, 'SparseToDense_updateGradInput')(
            ctx.spatial_size,
            ctx.input_metadata.ffi,
            ctx.input_features,
            grad_input.data,
            grad_output.data,
            torch.cuda.IntTensor() if ctx.input_features.is_cuda else nullptr)
        return grad_input, None, None, None, None
class SparseToDense(Module):
    def __init__(self, dimension, nPlanes):
        Module.__init__(self)
        self.dimension = dimension
        self.nPlanes=nPlanes

    def forward(self, input):
        return SparseToDenseFunction().apply(input.features,input.metadata,input.spatial_size,self.dimension,self.nPlanes)

    def input_spatial_size(self, out_size):
        return out_size

    def __repr__(self):
        return 'SparseToDense(' + str(self.dimension) + ','+ str(self.nPlanes)+ ')'
