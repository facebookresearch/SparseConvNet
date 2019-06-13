# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class BatchNormalization(Module):
    """
    Parameters:
    nPlanes : number of input planes
    eps : small number used to stabilise standard deviation calculation
    momentum : for calculating running average for testing (default 0.9)
    affine : only 'true' is supported at present (default 'true')
    noise : add multiplicative and additive noise during training if >0.
    leakiness : Apply activation def inplace: 0<=leakiness<=1.
    0 for ReLU, values in (0,1) for LeakyReLU, 1 for no activation def.
    """
    def __init__(
            self,
            nPlanes,
            eps=1e-4,
            momentum=0.9,
            affine=True,
            leakiness=1):
        Module.__init__(self)
        self.nPlanes = nPlanes
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.leakiness = leakiness
        self.register_buffer("running_mean", torch.Tensor(nPlanes).fill_(0))
        self.register_buffer("running_var", torch.Tensor(nPlanes).fill_(1))
        if affine:
            self.weight = Parameter(torch.Tensor(nPlanes).fill_(1))
            self.bias = Parameter(torch.Tensor(nPlanes).fill_(0))

    def forward(self, input):
        if input.features.nelement() == 0:
            return input
        assert input.features.size(1) == self.nPlanes, (self.nPlanes, input.features.shape)
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = BatchNormalizationFunction.apply(
            input.features,
            optionalTensor(self, 'weight'),
            optionalTensor(self, 'bias'),
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training,
            self.leakiness)
        return output

    def input_spatial_size(self, out_size):
        return out_size

    def __repr__(self):
        s = 'BatchNorm(' + str(self.nPlanes) + ',eps=' + str(self.eps) + \
            ',momentum=' + str(self.momentum) + ',affine=' + str(self.affine)
        if self.leakiness > 0:
            s = s + ',leakiness=' + str(self.leakiness)
        s = s + ')'
        return s


class BatchNormReLU(BatchNormalization):
    def __init__(self, nPlanes, eps=1e-4, momentum=0.9):
        BatchNormalization.__init__(self, nPlanes, eps, momentum, True, 0)

    def __repr__(self):
        s = 'BatchNormReLU(' + str(self.nPlanes) + ',eps=' + str(self.eps) + \
            ',momentum=' + str(self.momentum) + ',affine=' + str(self.affine) + ')'
        return s


class BatchNormLeakyReLU(BatchNormalization):
    def __init__(self, nPlanes, eps=1e-4, momentum=0.9, leakiness=0.333):
        BatchNormalization.__init__(self, nPlanes, eps, momentum, True, leakiness)

    def __repr__(self):
        s = 'BatchNormLeakyReLU(' + str(self.nPlanes) + ',eps=' + str(self.eps) + \
            ',momentum=' + str(self.momentum) + ',affine=' + str(self.affine) + ',leakiness='+str(self.leakiness)+')'
        return s

class BatchNormalizationFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            train,
            leakiness):
        ctx.nPlanes = running_mean.shape[0]
        ctx.train = train
        ctx.leakiness = leakiness
        output_features = input_features.new()
        saveMean = input_features.new().resize_(ctx.nPlanes)
        saveInvStd = running_mean.clone().resize_(ctx.nPlanes)
        sparseconvnet.SCN.BatchNormalization_updateOutput(
            input_features,
            output_features,
            saveMean,
            saveInvStd,
            running_mean,
            running_var,
            weight,
            bias,
            eps,
            momentum,
            ctx.train,
            ctx.leakiness)
        ctx.save_for_backward(input_features,
                              output_features,
                              weight,
                              bias,
                              running_mean,
                              running_var,
                              saveMean,
                              saveInvStd)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        input_features,\
            output_features,\
            weight,\
            bias,\
            running_mean,\
            running_var,\
            saveMean,\
            saveInvStd = ctx.saved_tensors
        assert ctx.train
        grad_input = grad_output.new()
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        sparseconvnet.SCN.BatchNormalization_backward(
            input_features,
            grad_input,
            output_features,
            grad_output.contiguous(),
            saveMean,
            saveInvStd,
            running_mean,
            running_var,
            weight,
            bias,
            grad_weight,
            grad_bias,
            ctx.leakiness)
        return grad_input, optionalTensorReturn(grad_weight), optionalTensorReturn(grad_bias), None, None, None, None, None, None

class MeanOnlyBNLeakyReLU(Module):
    """
    Parameters:
    nPlanes : number of input planes
    momentum : for calculating running average for testing (default 0.9)
    leakiness : Apply activation def inplace: 0<=leakiness<=1.
    0 for ReLU, values in (0,1) for LeakyReLU, 1 for no activation def.
    """
    def __init__(
            self,
            nPlanes,
            affine=True,
            leakiness=1,
            momentum=0.9):
        Module.__init__(self)
        self.nPlanes = nPlanes
        self.momentum = momentum
        self.register_buffer("running_mean", torch.Tensor(nPlanes).fill_(0))
        self.affine = affine
        if affine:
            self.bias = Parameter(torch.Tensor(nPlanes).fill_(0))
        self.leakiness = leakiness
    def forward(self, input):
        assert input.features.nelement() == 0 or input.features.size(1) == self.nPlanes
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        if self.training:
            with torch.no_grad():
                m=input.features.mean(0)
                self.running_mean=self.running_mean*self.momentum+m*(1-self.momentum)
            output.features = input.features - m
        else:
            output.features = input.features - self.running_mean
        if self.affine:
            output_features = output_features + self.bias
        if self.leakiness != 1:
            output.features = torch.nn.functional.leaky_relu(output.features, self.leakiness)
        return output

    def input_spatial_size(self, out_size):
        return out_size

    def __repr__(self):
        s = 'MeanOnlyBatchNorm(' + str(self.nPlanes) + ',momentum=' + str(self.momentum) + ',leakiness=' + str(self.leakiness) + ')'
        return s
