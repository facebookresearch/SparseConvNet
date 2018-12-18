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


class NetworkInNetworkFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias):
        output_features = input_features.new()
        ctx.save_for_backward(input_features,
                              output_features,
                              weight,
                              bias)
        sparseconvnet.forward_pass_multiplyAdd_count +=\
            sparseconvnet.SCN.NetworkInNetwork_updateOutput(
                input_features,
                output_features,
                weight,
                bias)
        sparseconvnet.forward_pass_hidden_states += output_features.nelement()
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features,\
            output_features,\
            weight,\
            bias = ctx.saved_tensors
        grad_input = grad_output.new()
        grad_weight = torch.zeros_like(weight)
        if bias is None:
            grad_bias = None
        else:
            grad_bias = torch.zeros_like(bias)
        sparseconvnet.SCN.NetworkInNetwork_updateGradInput(
            grad_input,
            grad_output,
            weight)
        sparseconvnet.SCN.NetworkInNetwork_accGradParameters(
            input_features,
            grad_output,
            grad_weight,
            grad_bias)
        return grad_input, grad_weight, grad_bias


class NetworkInNetwork(Module):
    def __init__(self, nIn, nOut, bias):
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

    def forward(self, input):
        assert input.features.nelement() == 0 or input.features.size(1) == self.nIn, (self.nIn, input.features.shape)
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = NetworkInNetworkFunction.apply(
            input.features,
            self.weight,
            optionalTensor(self, 'bias'))
        return output

    def __repr__(self):
        s = 'NetworkInNetwork' + str(self.nIn) + '->' + str(self.nOut)
        return s

    def input_spatial_size(self, out_size):
        return out_size
