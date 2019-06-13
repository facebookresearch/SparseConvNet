# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet
import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class SubmanifoldConvolution(Module):
    def __init__(self, dimension, nIn, nOut, filter_size, bias, groups=1):
        Module.__init__(self)
        self.dimension = dimension
        self.groups = groups
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = toLongTensor(dimension, filter_size)
        self.filter_volume = self.filter_size.prod().item()
        std = (2.0 * groups / nIn / self.filter_volume)**0.5
        self.weight = Parameter(torch.Tensor(
            self.filter_volume, groups, nIn//groups, nOut//groups
        ).normal_(0, std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())

    def forward(self, input):
        assert input.features.nelement() == 0 or input.features.size(1) == self.nIn, (self.nIn, self.nOut, input)
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = SubmanifoldConvolutionFunction.apply(
            input.features,
            self.weight,
            optionalTensor(self, 'bias'),
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
            sparseconvnet.SCN.SubmanifoldConvolution_updateOutput(
                spatial_size,
                filter_size,
                input_metadata,
                input_features,
                output_features,
                weight,
                bias)
        sparseconvnet.forward_pass_hidden_states += output_features.nelement()
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
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
