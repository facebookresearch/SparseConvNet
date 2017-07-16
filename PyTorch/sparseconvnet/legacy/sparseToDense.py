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

import torch
from . import SparseModule
from ..utils import dim_typed_fn, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class SparseToDense(SparseModule):
    def __init__(self, dimension):
        SparseModule.__init__(self)
        self.dimension = dimension
        self.output = torch.Tensor()
        self.gradInput = torch.FloatTensor()

    def updateOutput(self, input):
        dim_typed_fn(self.dimension, input, 'SparseToDense_updateOutput')(
            input.spatial_size,
            input.metadata.ffi,
            input.features,
            self.output,
            torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        return self.output

    def updateGradInput(self, input, gradOutput):
        dim_typed_fn(self.dimension, input, 'SparseToDense_updateGradInput')(
            input.spatial_size,
            input.metadata.ffi,
            input.features,
            self.gradInput,
            gradOutput,
            torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        return self.gradInput

    def __repr__(self):
        return 'SparseToDense(' + str(self.dimension) + ')'
