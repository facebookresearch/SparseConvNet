# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Fixed weight submanifold convolution - ineffcieit implementation
# prod(filter_size)* nIn outputs
# weight format locations x nInput x nOutput

import sparseconvnet
import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class ShapeContext(Module):
    def __init__(self, dimension, nIn, filter_size=3):
        Module.__init__(self)
        self.dimension = dimension
        self.filter_size = toLongTensor(dimension, filter_size)
        self.filter_volume = self.filter_size.prod().item()
        self.nIn = nIn
        self.nOut = nIn * self.filter_volume
        self.register_buffer("weight",
                             torch.eye(self.nOut).view(self.filter_volume, self.nIn, self.nOut))

    def forward(self, input):
        assert input.features.nelement() == 0 or input.features.size(1) == self.nIn, (self.nIn, self.nOut, input)
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = ShapeContextFunction.apply(
            input.features,
            self.weight,
            optionalTensor(self, 'bias'),
            input.metadata,
            input.spatial_size,
            self.dimension,
            self.filter_size)
        return output

    def __repr__(self):
        s = 'ShapeContext ' + \
            str(self.nIn) + '->' + str(self.nOut) + ' C'
        if self.filter_size.max() == self.filter_size.min():
            s = s + str(self.filter_size[0].item())
        else:
            s = s + '(' + str(self.filter_size[0].item())
            for i in self.filter_size[1:]:
                s = s + ',' + str(i.item())
            s = s + ')'
        return s

    def input_spatial_size(self, out_size):
        return out_size


class ShapeContextFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias,
            input_metadata,
            spatial_size,
            dimension,
            filter_size):
        ctx.input_metadata = input_metadata
        ctx.dimension = dimension
        output_features = input_features.new()
        ctx.save_for_backward(
            input_features,
            spatial_size,
            weight,
            bias,
            filter_size)

        sparseconvnet.SCN.SubmanifoldConvolution_updateOutput(
            spatial_size,
            filter_size,
            input_metadata,
            input_features,
            output_features,
            weight,
            bias)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "Don't backprop through ShapeContext!"
        input_features, spatial_size, weight, bias, filter_size = ctx.saved_tensors
        grad_input = grad_output.new()
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        sparseconvnet.SCN.SubmanifoldConvolution_backward(
            spatial_size,
            filter_size,
            ctx.input_metadata,
            input_features,
            grad_input,
            grad_output.contiguous(),
            weight,
            grad_weight,
            grad_bias)
        return grad_input, grad_weight, optionalTensorReturn(grad_bias), None, None, None, None

def MultiscaleShapeContext(dimension,n_features=1,n_layers=3,shape_context_size=3,downsample_size=2,downsample_stride=2,bn=True):
    m=sparseconvnet.Sequential()
    if n_layers==1:
        m.add(sparseconvnet.ShapeContext(dimension,n_features,shape_context_size))
    else:
        m.add(
            sparseconvnet.ConcatTable().add(
                sparseconvnet.ShapeContext(dimension, n_features, shape_context_size)).add(
                sparseconvnet.Sequential(
                    sparseconvnet.AveragePooling(dimension,downsample_size,downsample_stride),
                    MultiscaleShapeContext(dimension,n_features,n_layers-1,shape_context_size,downsample_size,downsample_stride,False),
                    sparseconvnet.UnPooling(dimension,downsample_size,downsample_stride)))).add(
            sparseconvnet.JoinTable())
    if bn:
        m.add(sparseconvnet.BatchNormalization(shape_context_size**dimension*n_features*n_layers))
    return m
