# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet
from torch.autograd import Function, Variable
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class NetworkInNetworkFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features,
        weight,
        bias):
        ctx.input_features=input_features
        ctx.weight=weight
        ctx.bias=bias
        ctx.output_features=input_features.new()
        sparseconvnet.forward_pass_multiplyAdd_count +=\
            typed_fn(input_features, 'NetworkInNetwork_updateOutput')(
                input_features,
                ctx.output_features,
                weight,
                bias if bias is not None else nullptr)
        sparseconvnet.forward_pass_hidden_states += ctx.output_features.nelement()
        return ctx.output_features
    @staticmethod
    def backward(ctx, grad_output):
        grad_input=Variable(grad_output.data.new())
        grad_weight=Variable(grad_output.data.new().resize_as_(ctx.weight).zero())
        if ctx.bias is None:
            grad_bias=None
        else:
            grad_bias = Variable(grad_output.data.new().resize_as_(ctx.bias))
        typed_fn(ctx.input_features, 'NetworkInNetwork_updateGradInput')(
            grad_input.data,
            grad_output.data,
            ctx.weight)
        typed_fn(ctx.input_features, 'NetworkInNetwork_accGradParameters')(
            ctx.input_features,
            grad_output.data,
            grad_weight.data,
            grad_bias.data if grad_bias is not None else nullptr)
        return grad_input, grad_weight, grad_bias

class NetworkInNetwork(Module):
    def __init__(self, nIn, nOut, bias=False):
        Module.__init__(self)
        self.nIn = nIn
        self.nOut = nOut
        std = (2.0 / nIn)**0.5
        self.weight = Parameter(torch.Tensor(
            nIn, nOut).normal_(
            0,
            std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())
        else:
            self.bias=None
    def forward(self, input):
        assert input.features.ndimension()==0 or input.features.size(1) == self.nIn
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features=NetworkInNetworkFunction().apply(
            input.features,
            self.weight,
            self.bias)
        return output

    def __repr__(self):
        s = 'NetworkInNetwork' + str(self.nIn) + '->' + str(self.nOut)
        return s

    def input_spatial_size(self, out_size):
        return out_size
