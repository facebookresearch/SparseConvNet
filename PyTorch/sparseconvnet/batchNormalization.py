# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

from torch.autograd import Function, Variable
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor

class BatchNormalizationFunction(Function):
    @staticmethod
    def forward(
            ctx,
            input_features,
            weight,
            bias,
            runningMean,
            runningVar,
            eps,
            momentum,
            train,
            leakiness):
        ctx.nPlanes=runningMean.shape[0]
        ctx.input_features=input_features
        ctx.weight=weight
        ctx.bias=bias
        ctx.runningMean=runningMean
        ctx.runningVar=runningVar
        ctx.train=train
        ctx.leakiness=leakiness
        ctx.output_features = input_features.new()
        ctx.saveMean = input_features.new().resize_(ctx.nPlanes)
        ctx.saveInvStd = runningMean.clone().resize_(ctx.nPlanes)
        typed_fn(input_features, 'BatchNormalization_updateOutput')(
            input_features,
            ctx.output_features,
            ctx.saveMean,
            ctx.saveInvStd,
            ctx.runningMean,
            ctx.runningVar,
            ctx.weight if ctx.weight is not None else nullptr,
            ctx.bias if ctx.bias is not None else nullptr,
            eps,
            momentum,
            ctx.train,
            ctx.leakiness)
        return ctx.output_features

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.train
        grad_input=Variable(grad_output.data.new())
        if ctx.weight is None:
            grad_weight=None
        else:
            grad_weight=Variable(ctx.input_features.new().resize_(ctx.nPlanes).zero_())
        if ctx.bias is None:
            grad_bias=None
        else:
            grad_bias=Variable(ctx.input_features.new().resize_(ctx.nPlanes).zero_())
        typed_fn(ctx.input_features, 'BatchNormalization_backward')(
            ctx.input_features,
            grad_input.data,
            ctx.output_features,
            grad_output.data.contiguous(),
            ctx.saveMean,
            ctx.saveInvStd,
            ctx.runningMean,
            ctx.runningVar,
            ctx.weight if ctx.weight is not None else nullptr,
            ctx.bias if ctx.bias is not None else nullptr,
            grad_weight.data if grad_weight is not None else nullptr,
            grad_bias.data if grad_bias is not None else nullptr,
            ctx.leakiness)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class BatchNormalization(Module):
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
        self.register_buffer("runningMean", torch.Tensor(nPlanes).fill_(0))
        self.register_buffer("runningVar", torch.Tensor(nPlanes).fill_(1))
        if affine:
            self.weight = Parameter(torch.Tensor(nPlanes).fill_(1))
            self.bias = Parameter(torch.Tensor(nPlanes).fill_(0))
        else:
            self.weight = None
            self.bias = None
    def forward(self, input):
        assert input.features.ndimension()==0 or input.features.size(1) == self.nPlanes
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = BatchNormalizationFunction().apply(
            input.features,
            self.weight,
            self.bias,
            self.runningMean,
            self.runningVar,
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
    def __init__(self, nPlanes, eps=1e-4, momentum=0.9):
        BatchNormalization.__init__(self, nPlanes, eps, momentum, True, 0.333)
    def __repr__(self):
        s = 'BatchNormReLU(' + str(self.nPlanes) + ',eps=' + str(self.eps) + \
            ',momentum=' + str(self.momentum) + ',affine=' + str(self.affine) + ')'
        return s
