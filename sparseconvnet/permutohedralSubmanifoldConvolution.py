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

def permutohedral_basis(dimension):
    """
    Calculate two matrices: a, a_inverse
    Use torch.mm(coordinates, a_inverse) to map into permutohedral coordinates space, before input goes to SparseConvNet
    """
    a=torch.zeros(dimension,dimension)
    for i in range(dimension):
        for j in range(i):
            dp=(a[i,:]*a[j,:]).sum()
            a[i,j]=(0.5-dp)/a[j,j]
        dp=(a[i,:]*a[i,:]).sum()
        a[i,i]=(1-dp)**0.5
    ai=torch.inverse(a)
    return a, ai

class PermutohedralSubmanifoldConvolution(Module):
    def __init__(self, dimension, nIn, nOut, bias, groups=1):
        Module.__init__(self)
        self.dimension = dimension
        self.groups=groups
        self.nIn = nIn
        self.nOut = nOut
        self.filter_volume = dimension**2 + dimension + 1
        std = (2.0 / nIn / self.filter_volume)**0.5
        self.weight = Parameter(torch.Tensor(
            self.filter_volume, groups, nIn//groups, nOut//groups
        ).normal_(0, std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())

    def forward(self, input):
        assert input.features.nelement() == 0 or input.features.size(1) == self.nIn
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = PermutohedralSubmanifoldConvolutionFunction.apply(
            input.features,
            self.weight,
            optionalTensor(self, 'bias'),
            input.metadata,
            input.spatial_size,
            self.dimension)
        return output

    def __repr__(self):
        s = 'PermutohedralSubmanifoldConvolution'
        return s

    def input_spatial_size(self, out_size):
        return out_size


class PermutohedralSubmanifoldConvolutionFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias,
            input_metadata,
            spatial_size,
            dimension):
        ctx.input_metadata = input_metadata
        ctx.dimension = dimension
        output_features = input_features.new()
        ctx.save_for_backward(
            input_features,
            spatial_size,
            weight,
            bias)

        sparseconvnet.forward_pass_multiplyAdd_count +=\
            sparseconvnet.SCN.PermutohedralSubmanifoldConvolution_updateOutput(
                spatial_size,
                input_metadata,
                input_features,
                output_features,
                weight,
                bias)
        sparseconvnet.forward_pass_hidden_states += output_features.nelement()
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features, spatial_size, weight, bias = ctx.saved_tensors
        grad_input = grad_output.new()
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        sparseconvnet.SCN.PermutohedralSubmanifoldConvolution_backward(
            spatial_size,
            ctx.input_metadata,
            input_features,
            grad_input,
            grad_output.contiguous(),
            weight,
            grad_weight,
            grad_bias)
        return grad_input, grad_weight, optionalTensorReturn(grad_bias), None, None, None, None
