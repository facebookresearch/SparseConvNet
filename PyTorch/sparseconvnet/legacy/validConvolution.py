# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet as s
from . import SparseModule
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class ValidConvolution(SparseModule):
    def __init__(self, dimension, nIn, nOut, filter_size, bias):
        SparseModule.__init__(self)
        self.dimension = dimension
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = toLongTensor(dimension, filter_size)
        self.filter_volume = self.filter_size.prod()
        self.weight = torch.Tensor(
            nIn * self.filter_volume, nOut
        ).normal_(0, (2.0 / nIn / self.filter_volume)**0.5)
        self.gradWeight = torch.Tensor(nIn * self.filter_volume, nOut)
        if bias:
            self.bias = torch.Tensor(nOut).zero_()
            self.gradBias = torch.Tensor(nOut)
        self.output = SparseConvNetTensor(torch.Tensor())
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        assert input.features.size(1) == self.nIn
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        s.forward_pass_multiplyAdd_count +=\
            dim_typed_fn(self.dimension, input, 'ValidConvolution_updateOutput')(
                input.spatial_size,
                self.filter_size,
                input.metadata.ffi,
                input.features,
                self.output.features,
                self.weight,
                optionalTensor(self, 'bias'),
                self.filter_volume,
                torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        s.forward_pass_hidden_states += self.output.features.nelement()
        return self.output

    def backward(self, input, gradOutput, scale=1):
        assert scale == 1
        dim_typed_fn(self.dimension, input, 'ValidConvolution_backward')(
            input.spatial_size,
            self.filter_size,
            input.metadata.ffi,
            input.features,
            self.gradInput,
            gradOutput,
            self.weight,
            self.gradWeight,
            optionalTensor(self, 'gradBias'),
            self.filter_volume,
            torch.cuda.IntTensor() if input.features.is_cuda else nullptr)
        return self.gradInput

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

    def __repr__(self):
        s = 'ValidConvolution ' + str(self.nIn) + '->' + str(self.nOut) + ' C'
        if self.filter_size.max() == self.filter_size.min():
            s = s + str(self.filter_size[0])
        else:
            s = s + '(' + str(self.filter_size[0])
            for i in self.filter_size[1:]:
                s = s + ',' + str(i)
            s = s + ')'
        return s
