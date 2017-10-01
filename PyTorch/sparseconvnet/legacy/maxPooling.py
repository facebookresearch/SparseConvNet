# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr
from ..sparseConvNetTensor import SparseConvNetTensor


class MaxPooling(SparseModule):
    def __init__(self, dimension, pool_size, pool_stride, nFeaturesToDrop=0):
        SparseModule.__init__(self)
        self.dimension = dimension
        self.pool_size = toLongTensor(dimension, pool_size)
        self.pool_stride = toLongTensor(dimension, pool_stride)
        self.pool_volume = self.pool_size.prod()
        self.nFeaturesToDrop = nFeaturesToDrop or 0
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size =\
            (input.spatial_size - self.pool_size) / self.pool_stride + 1
        dim_typed_fn(self.dimension, input.features, 'MaxPooling_updateOutput')(
            input.spatial_size,
            self.output.spatial_size,
            self.pool_size,
            self.pool_stride,
            input.metadata.ffi,
            input.features,
            self.output.features,
            self.nFeaturesToDrop,
            torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        return self.output

    def updateGradInput(self, input, gradOutput):
        dim_typed_fn(self.dimension, input.features, 'MaxPooling_updateGradInput')(
            input.spatial_size,
            self.output.spatial_size,
            self.pool_size,
            self.pool_stride,
            input.metadata.ffi,
            input.features,
            self.gradInput,
            self.output.features,
            gradOutput,
            self.nFeaturesToDrop,
            torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        return self.gradInput

    def type(self, t=None, tensorCache=None):
        if t is None:
            return self._type
        self.output.type(t)
        self.gradInput = self.gradInput.type(t)

    def __repr__(self):
        s = 'MaxPooling'
        if self.pool_size.max() == self.pool_size.min() and\
                self.pool_stride.max() == self.pool_stride.min():
            s = s + str(self.pool_size[0]) + '/' + str(self.pool_stride[0])
        else:
            s = s + '(' + str(self.pool_size[0])
            for i in self.pool_size[1:]:
                s = s + ',' + str(i)
            s = s + ')/(' + str(self.pool_stride[0])
            for i in self.pool_stride[1:]:
                s = s + ',' + str(i)
            s = s + ')'

        if self.nFeaturesToDrop > 0:
            s = s + ' nFeaturesToDrop = ' + self.nFeaturesToDrop
        return s

    def suggestInputSize(self, out_size):
        return (out_size - 1) * self.pool_stride + self.pool_size
