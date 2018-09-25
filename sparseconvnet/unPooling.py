# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet.SCN
from torch.autograd import Function, Variable
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class UnPoolingFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            input_metadata,
            input_spatial_size,
            output_spatial_size,
            dimension,
            pool_size,
            pool_stride,
            nFeaturesToDrop):
        ctx.input_features=input_features
        ctx.input_metadata=input_metadata
        ctx.input_spatial_size = input_spatial_size
        ctx.output_spatial_size = output_spatial_size
        ctx.dimension = dimension
        ctx.pool_size = pool_size
        ctx.pool_stride = pool_stride
        ctx.nFeaturesToDrop = nFeaturesToDrop
        output_features = input_features.new()
        sparseconvnet.SCN.UnPooling_updateOutput(
            input_spatial_size,
            output_spatial_size,
            pool_size,
            pool_stride,
            input_metadata,
            input_features,
            output_features,
            nFeaturesToDrop)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input=Variable(grad_output.data.new())
        sparseconvnet.SCN.UnPooling_updateGradInput(
            ctx.input_spatial_size,
            ctx.output_spatial_size,
            ctx.pool_size,
            ctx.pool_stride,
            ctx.input_metadata,
            ctx.input_features,
            grad_input.data,
            grad_output.data.contiguous(),
            ctx.nFeaturesToDrop)
        return grad_input, None, None, None, None, None, None, None


class UnPooling(Module):
    def __init__(self, dimension, pool_size, pool_stride, nFeaturesToDrop=0):
        super(UnPooling, self).__init__()
        self.dimension = dimension
        self.pool_size = toLongTensor(dimension, pool_size)
        self.pool_stride = toLongTensor(dimension, pool_stride)
        self.nFeaturesToDrop = nFeaturesToDrop
    def forward(self, input):
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size =\
            (input.spatial_size - 1) * self.pool_stride + self.pool_size
        output.features =  UnPoolingFunction().apply(
            input.features, input.metadata, input.spatial_size,
            output.spatial_size, self.dimension,self.pool_size,self.pool_stride,
            self.nFeaturesToDrop)
        return output
    def input_spatial_size(self, out_size):
        return (out_size - 1) * self.pool_stride + self.pool_size
    def __repr__(self):
        s = 'UnPooling'
        if self.pool_size.max() == self.pool_size.min() and\
                self.pool_stride.max() == self.pool_stride.min():
            s = s + str(self.pool_size[0].item()) + '/' + str(self.pool_stride[0].item())
        else:
            s = s + '(' + str(self.pool_size[0].item())
            for i in self.pool_size[1:]:
                s = s + ',' + str(i)
            s = s + ')/(' + str(self.pool_stride[0].item())
            for i in self.pool_stride[1:]:
                s = s + ',' + str(i)
            s = s + ')'

        if self.nFeaturesToDrop > 0:
            s = s + ' nFeaturesToDrop = ' + self.nFeaturesToDrop
        return s
