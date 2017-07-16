# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
  Affine transformation (i.e. the second half of a typical batchnormalization layer)
  Parameters:
  nPlanes : number of input planes
  noise : add multiplicative and additive noise during training if >0.
  leakiness : Apply activation function inplace: 0<=leakiness<=1.
  0 for ReLU, values in (0,1) for LeakyReLU, 1 for no activation function.
"""

import torch
import sparseconvnet as s
from . import SparseModule
from ..utils import toLongTensor, typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor
import math


class AffineReLUTrivialConvolution(SparseModule):
    def __init__(self, nIn, nOut, additiveGrad=False):
        SparseModule.__init__(self)
        self.nIn = nIn
        self.nOut = nOut
        self.affineWeight = torch.Tensor(nIn).fill_(1)
        self.affineBias = torch.Tensor(nIn).zero_()
        self.convWeight = torch.Tensor(
            nIn, nOut).normal_(
            0, math.sqrt(
                2.0 / nIn))
        self.gradAffineWeight = torch.Tensor(nIn).fill_(0)
        self.gradAffineBias = torch.Tensor(nIn).zero_()
        self.gradConvWeight = torch.Tensor(nIn, nOut).zero_()
        self.additiveGrad = additiveGrad
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = torch.Tensor()

    def parameters(self):
        return [self.affineWeight, self.affineBias, self.convWeight], [
            self.gradAffineWeight, self.gradAffineBias, self.gradConvWeight]

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        typed_fn(input, 'AffineReluTrivialConvolution_updateOutput')(
            input.features,
            self.output.features,
            self.affineWeight,
            self.affineBias,
            self.convWeight)
        s.forward_pass_multiplyAdd_count += input.features.size(
            0) * self.nIn * self.nOut
        s.forward_pass_hidden_states += self.output.features.nelement()
        return self.output

    def backward(self, input, gradOutput, scale=1):
        assert scale == 1
        typed_fn(input, 'AffineReluTrivialConvolution_backward')(
            input.features,
            self.gradInput,
            gradOutput,
            self.affineWeight,
            self.gradAffineWeight,
            self.affineBias,
            self.gradAffineBias,
            self.convWeight,
            self.gradConvWeight,
            self.additiveGrad)
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        assert false  # just call backward

    def accGradParameters(input, gradOutput, scale):
        assert false  # just call backward

    def __repr__(self):
        s = 'AffineReluTrivialConvolution ' + \
            str(self.nIn) + '->' + str(self.nOut)
        return s

    def type(self, t=None, tensorCache=None):
        if t is None:
            return self._type
        self._type = t
        self.affineWeight = self.affineWeight.type(t)
        self.affineBias = self.affineBias.type(t)
        self.convWeight = self.convWeight.type(t)
        self.gradAffineWeight = self.gradAffineWeight.type(t)
        self.gradAffineBias = self.gradAffineBias.type(t)
        self.gradConvWeight = self.gradConvWeight.type(t)
        self.gradInput = self.gradInput.type(t)
        self.output.type(t)
