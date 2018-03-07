# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 'SubmanifoldConvolution == SubmanifoldConvolution'

import sparseconvnet
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class SubmanifoldConvolution(Module):
    def __init__(self, dimension, nIn, nOut, filter_size, bias):
        Module.__init__(self)
        self.dimension = dimension
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = toLongTensor(dimension, filter_size)
        self.filter_volume = self.filter_size.prod().item()
        std = (2.0 / nIn / self.filter_volume)**0.5
        self.weight = Parameter(torch.Tensor(
            nIn * self.filter_volume, nOut
        ).normal_(0, std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())
        else:
            self.bias = None

    def forward(self, input):
        assert input.features.ndimension() == 0 or input.features.size(1) == self.nIn
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = SubmanifoldConvolutionFunction.apply(
            input.features,
            self.weight,
            self.bias,
            input.metadata,
            input.spatial_size,
            self.dimension,
            self.filter_size)
        return output

    def __repr__(self):
        s = 'SubmanifoldConvolution ' + \
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


class ValidConvolution(SubmanifoldConvolution):
    pass

class SubmanifoldConvolutionFunction(Function):
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
        sparseconvnet.forward_pass_multiplyAdd_count +=\
            dim_typed_fn(
                dimension, input_features, 'SubmanifoldConvolution_updateOutput')(
                spatial_size,
                filter_size,
                input_metadata.ffi,
                input_features.data,
                output_features.data,
                weight.data,
                bias.data if bias is not None else nullptr,
                0,  # remove this parameter!!
                torch.cuda.IntTensor() if input_features.is_cuda else nullptr)
        sparseconvnet.forward_pass_hidden_states += output_features.nelement()
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features, spatial_size, weight, bias, filter_size = ctx.saved_tensors
        grad_input = grad_output.new()
        grad_weight = grad_output.new().resize_as_(weight).zero_()
        if bias is None:
            grad_bias = None
        else:
            grad_bias = grad_output.new().resize_as_(bias).zero_()
        dim_typed_fn(
            ctx.dimension, input_features, 'SubmanifoldConvolution_backward')(
            spatial_size,
            filter_size,
            ctx.input_metadata.ffi,
            input_features,
            grad_input,
            grad_output.contiguous(),
            weight,
            grad_weight,
            grad_bias.data if grad_bias is not None else nullptr,
            0,  # remove this parameter
            torch.cuda.IntTensor() if input_features.is_cuda else nullptr)
        return grad_input, grad_weight, grad_bias, None, None, None, None
