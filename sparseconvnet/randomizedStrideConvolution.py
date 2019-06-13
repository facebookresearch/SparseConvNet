# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet, sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .convolution import ConvolutionFunction

class RandomizedStrideConvolution(Module):
    """
    A bit like Fractional Max Pooling during training, but at test time it
    operates like a regular sparseconvnet.Convolution.

    Don't use with scn.InputBatch.precomute_metadata()!!

    Like regular a sparseconvnet.Convolution, is can be paired with a
    sparseconvnet.Deconvolution module in an UNet style network, to restore
    the input sparsity pattern.
    """
    def __init__(self, dimension, nIn, nOut, filter_size, filter_stride, bias, groups=1):
        Module.__init__(self)
        self.dimension = dimension
        self.groups = groups
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = toLongTensor(dimension, filter_size)
        self.filter_volume = self.filter_size.prod().item()
        self.filter_stride = toLongTensor(dimension, filter_stride)
        std = (2.0 * groups / nIn / self.filter_volume)**0.5
        self.weight = Parameter(torch.Tensor(
            self.filter_volume, groups, nIn//groups, nOut//groups).normal_(
            0,
            std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())

    def forward(self, input):
        assert input.features.ndimension() == 0 or input.features.size(1) == self.nIn
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size =\
            (input.spatial_size - self.filter_size) / self.filter_stride + 1
        assert ((output.spatial_size - 1) * self.filter_stride +
                self.filter_size == input.spatial_size).all()
        output.features = (RandomizedStrideConvolutionFunction if self.training else ConvolutionFunction).apply(
        #output.features = ConvolutionFunction.apply(
        #output.features = RandomizedStrideConvolutionFunction.apply(
            input.features,
            self.weight,
            optionalTensor(self, 'bias'),
            input.metadata,
            input.spatial_size,
            output.spatial_size,
            self.dimension,
            self.filter_size,
            self.filter_stride)
        return output

    def __repr__(self):
        s = 'RandomizedStrideConvolution ' + str(self.nIn) + '->' + str(self.nOut) + ' C'
        if self.filter_size.max().item() == self.filter_size.min().item() and\
                self.filter_stride.max().item() == self.filter_stride.min().item():
            s = s + str(self.filter_size[0].item()) + \
                '/' + str(self.filter_stride[0].item())
        else:
            s = s + '(' + str(self.filter_size[0].item())
            for i in self.filter_size[1:]:
                s = s + ',' + str(i.item())
            s = s + ')/(' + str(self.filter_stride[0].item())
            for i in self.filter_stride[1:]:
                s = s + ',' + str(i.item())
            s = s + ')'
        return s

    def input_spatial_size(self, out_size):
        return (out_size - 1) * self.filter_stride + self.filter_size

class RandomizedStrideConvolutionFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias,
            input_metadata,
            input_spatial_size,
            output_spatial_size,
            dimension,
            filter_size,
            filter_stride):
        output_features = input_features.new()
        ctx.input_metadata = input_metadata
        ctx.dimension = dimension
        ctx.save_for_backward(
            input_features,
            input_spatial_size,
            weight,
            bias,
            output_spatial_size,
            filter_size,
            filter_stride)
        sparseconvnet.forward_pass_multiplyAdd_count +=\
            sparseconvnet.SCN.RandomizedStrideConvolution_updateOutput(
                input_spatial_size,
                output_spatial_size,
                filter_size,
                filter_stride,
                input_metadata,
                input_features,
                output_features,
                weight,
                bias)

        sparseconvnet.forward_pass_hidden_states += output_features.nelement()
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features, input_spatial_size, weight, bias, output_spatial_size, filter_size, filter_stride = ctx.saved_tensors
        grad_input = grad_output.new()
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        sparseconvnet.SCN.RandomizedStrideConvolution_backward(
            input_spatial_size,
            output_spatial_size,
            filter_size,
            filter_stride,
            ctx.input_metadata,
            input_features,
            grad_input,
            grad_output.contiguous(),
            weight,
            grad_weight,
            grad_bias)
        return grad_input, grad_weight, optionalTensorReturn(grad_bias), None, None, None, None, None, None
