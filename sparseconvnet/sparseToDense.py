# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
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

import sparseconvnet.SCN
from torch.autograd import Function
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
        ctx.input_metadata = input_metadata
        ctx.dimension = dimension
        ctx.save_for_backward(input_features, spatial_size)
        output = input_features.new()
        sparseconvnet.SCN.SparseToDense_updateOutput(
            spatial_size,
            input_metadata,
            input_features,
            output,
            nPlanes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        input_features, spatial_size = ctx.saved_tensors
        sparseconvnet.SCN.SparseToDense_updateGradInput(
            spatial_size,
            ctx.input_metadata,
            input_features,
            grad_input,
            grad_output.contiguous())
        return grad_input, None, None, None, None


class SparseToDense(Module):
    def __init__(self, dimension, nPlanes):
        Module.__init__(self)
        self.dimension = dimension
        self.nPlanes = nPlanes

    def forward(self, input):
        return SparseToDenseFunction.apply(
            input.features,
            input.metadata,
            input.spatial_size,
            self.dimension,
            self.nPlanes)

    def input_spatial_size(self, out_size):
        return out_size

    def __repr__(self):
        return 'SparseToDense(' + str(self.dimension) + \
            ',' + str(self.nPlanes) + ')'
