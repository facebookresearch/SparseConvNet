# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from . import SparseModule
import sparseconvnet as s
from ..utils import toLongTensor, typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class NetworkInNetwork(SparseModule):
    def __init__(self, nIn, nOut, bias=True):
        SparseModule.__init__(self)
        self.nIn = nIn
        self.nOut = nOut
        self.weight = torch.Tensor(nIn, nOut).normal_(0, (2.0 / self.nIn)**0.5)
        self.gradWeight = torch.Tensor(nIn, nOut)
        if bias:
            self.bias = torch.Tensor(nOut).fill_(0)
            self.gradBias = torch.Tensor(nOut)
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        s.forward_pass_multiplyAdd_count +=\
            typed_fn(input, 'NetworkInNetwork_updateOutput')(
                input.features,
                self.output.features,
                self.weight,
                optionalTensor(self, 'bias'))
        s.forward_pass_hidden_states += self.output.features.nelement()
        return self.output

    def updateGradInput(self, input, gradOutput):
        typed_fn(input, 'NetworkInNetwork_updateGradInput')(
            self.gradInput,
            gradOutput,
            self.weight)
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        assert scale == 1
        typed_fn(input, 'NetworkInNetwork_accGradParameters')(
            input.features,
            gradOutput,
            self.gradWeight,
            optionalTensor(self, 'gradBias'))

    def __repr__(self):
        s = 'NetworkInNetwork' + str(self.nIn) + '->' + str(self.nOut)
        return s

    def type(self, t=None, tensorCache=None):
        if t is None:
            return self._type
        self._type = t
        self.weight = self.weight.type(t)
        self.gradWeight = self.gradWeight.type(t)
        self.output.type(t)
        self.gradInput = self.gradInput.type(t)
        if hasattr(self, 'bias'):
            self.bias = self.bias.type(t)
            self.gradBias = self.gradBias.type(t)
