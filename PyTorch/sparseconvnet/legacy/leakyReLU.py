# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class LeakyReLU(SparseModule):
    def __init__(self, leakage=0.333, ip=True):
        SparseModule.__init__(self)
        self.inplace = ip
        self.leakage = leakage
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = None if ip else torch.Tensor()

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        typed_fn(input.features, 'LeakyReLU_updateOutput')(
            input.features,
            self.output.features,
            self.leakage)
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput = gradOutput
        typed_fn(input.features, 'LeakyReLU_updateGradInput')(
            input.features,
            self.gradInput,
            gradOutput,
            self.leakage)
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t:
            self.output.type(t)
            self.gradInput = self.gradInput.type(t)


class Tanh(SparseModule):
    def __init__(self):
        SparseModule.__init__(self)
        self.output = SparseConvNetTensor(torch.Tensor())
        #self.gradInput = None if ip else torch.Tensor()
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        self.output.features=torch.tanh(input.features)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(gradOutput).copy_(gradOutput)
        self.gradInput.mul(1+self.output.features)
        self.gradInput.mul(1-self.output.features)
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t:
            self.output.type(t)
            self.gradInput = self.gradInput.type(t)
