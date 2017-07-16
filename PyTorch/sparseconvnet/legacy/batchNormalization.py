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

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class BatchNormalization(SparseModule):
    def __init__(
            self,
            nPlanes,
            eps=1e-4,
            momentum=0.9,
            affine=True,
            leakiness=1):
        SparseModule.__init__(self)
        assert nPlanes % 4 == 0
        self.nPlanes = nPlanes
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.leakiness = leakiness
        self.saveMean = torch.Tensor(nPlanes).fill_(0)
        self.saveInvStd = torch.Tensor(nPlanes).fill_(1)
        self.runningMean = torch.Tensor(nPlanes).fill_(0)
        self.runningVar = torch.Tensor(nPlanes).fill_(1)
        if affine:
            self.weight = torch.Tensor(nPlanes).fill_(1)
            self.bias = torch.Tensor(nPlanes).fill_(0)
            self.gradWeight = torch.Tensor(nPlanes)
            self.gradBias = torch.Tensor(nPlanes)
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        assert input.features.size(1) == self.nPlanes
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        typed_fn(input, 'BatchNormalization_updateOutput')(
            input.features,
            self.output.features,
            self.saveMean,
            self.saveInvStd,
            self.runningMean,
            self.runningVar,
            optionalTensor(self, 'weight'),
            optionalTensor(self, 'bias'),
            self.eps,
            self.momentum,
            self.train,
            self.leakiness)
        return self.output

    def backward(self, input, gradOutput, scale=1):
        assert scale == 1
        assert self.train
        typed_fn(input, 'BatchNormalization_backward')(
            input.features,
            self.gradInput,
            self.output.features,
            gradOutput,
            self.saveMean,
            self.saveInvStd,
            self.runningMean,
            self.runningVar,
            optionalTensor(self, 'weight'),
            optionalTensor(self, 'bias'),
            optionalTensor(self, 'gradWeight'),
            optionalTensor(self, 'gradBias'),
            self.leakiness)
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        assert false  # just call backward

    def accGradParameters(self, input, gradOutput, scale):
        assert false  # just call backward

    def type(self, t=None, tensorCache=None):
        self.output.type(t)
        SparseModule.type(self, t, tensorCache)

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


class BatchNormalizationInTensor(BatchNormalization):
    def __init__(
            self,
            nPlanes,
            eps=1e-4,
            momentum=0.9,
            output_column_offset=0):
        BatchNormalization.__init__(self, nPlanes, eps, momentum, False, 1)
        self.output_column_offset = output_column_offset

    def updateOutput(self, input):
        o = self.output.features.narrow(
            1, self.output_column_offset, self.nPlanes)
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        typed_fn(input, 'BatchNormalizationInTensor_updateOutput')(
            input.features,
            o,
            self.saveMean,
            self.saveInvStd,
            self.runningMean,
            self.runningVar,
            optionalTensor(self, 'weight'),
            optionalTensor(self, 'bias'),
            self.eps,
            self.momentum,
            self.train,
            self.leakiness)
        return self.output

    def backward(self, input, gradOutput, scale=1):
        assert scale == 1
        assert self.train
        o = self.output.features.narrow(
            1, self.output_column_offset, self.nPlanes)
        d_o = gradOutput.narrow(1, self.output_column_offset, self.nPlanes)
        typed_fn(input, 'BatchNormalization_backward')(
            input.features,
            self.gradInput,
            o,
            d_o,
            self.saveMean,
            self.saveInvStd,
            self.runningMean,
            self.runningVar,
            optionalTensor(self, 'weight'),
            optionalTensor(self, 'bias'),
            optionalTensor(self, 'gradWeight'),
            optionalTensor(self, 'gradBias'),
            self.leakiness)
        return self.gradInput
